import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from server.core.config import settings
from server.services.file_service import FileService
from server.services.vision_service import VisionService
from server.services.svg_service import SVGService
from server.services.id_photo_service import IdPhotoService
from server.core.connection_manager import manager

router = APIRouter()

def cleanup_files(*file_paths: str):
    """Background task to remove temporary files"""
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error deleting temp file {path}: {e}")

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temp dir and return path"""
    suffix = os.path.splitext(upload_file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{suffix}"
    temp_path = settings.TEMP_DIR / temp_filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return str(temp_path)

@router.post("/pdf/compress")
async def compress_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ratio: float = Form(0.5),
    task_id: str = Form(None)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    input_path = await save_upload_file(file)
    output_filename = f"compressed_{file.filename}"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        # Callback for progress
        async def progress_callback(percent, message):
            if task_id:
                await manager.update_progress(task_id, percent, message)

        await FileService.compress_pdf(input_path, output_path, ratio, progress_callback)
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(output_path, filename=output_filename)
    except Exception as e:
        cleanup_files(input_path)
        if os.path.exists(output_path):
            cleanup_files(output_path)
        # If error, notify WS
        if task_id:
             await manager.update_progress(task_id, -1, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/convert")
async def convert_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_format: str = Form("PNG")
):
    input_path = await save_upload_file(file)
    
    # Determine output extension
    target_format = target_format.upper()
    ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "PDF": ".pdf", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tiff"}
    output_ext = ext_map.get(target_format, ".png")
    
    original_name = os.path.splitext(file.filename)[0]
    output_filename = f"{original_name}{output_ext}"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        FileService.convert_format(input_path, output_path, target_format)
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(output_path, filename=output_filename)
    except Exception as e:
        cleanup_files(input_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/rotate")
async def rotate_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    angle: float = Form(...)
):
    input_path = await save_upload_file(file)
    output_filename = f"rotated_{file.filename}"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        VisionService.rotate_image(input_path, output_path, angle)
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(output_path, filename=output_filename)
    except Exception as e:
        cleanup_files(input_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/resize")
async def resize_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    ratio: Optional[float] = Form(None)
):
    input_path = await save_upload_file(file)
    output_filename = f"resized_{file.filename}"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        if ratio:
            FileService.compress_image(input_path, output_path, ratio=ratio)
        else:
            VisionService.resize_image(input_path, output_path, width, height)
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(output_path, filename=output_filename)
    except Exception as e:
        cleanup_files(input_path)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================
# SVG转换端点
# ==============================================

@router.post("/svg/to-svg")
async def convert_to_svg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    trace_mode: str = Form("default")
):
    """
    多种格式转SVG
    支持: EPS, JPG, PNG, PDF
    """
    input_path = await save_upload_file(file)
    file_ext = os.path.splitext(file.filename)[1].lower()
    original_name = os.path.splitext(file.filename)[0]
    output_filename = f"{original_name}.svg"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        if file_ext == '.eps':
            SVGService.eps_to_svg(input_path, output_path)
        elif file_ext in ['.jpg', '.jpeg']:
            SVGService.jpg_to_svg(input_path, output_path, trace_mode)
        elif file_ext == '.png':
            SVGService.png_to_svg(input_path, output_path, trace_mode)
        elif file_ext == '.pdf':
            page_num = 0  # 默认第一页
            SVGService.pdf_to_svg(input_path, output_path, page_num)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(output_path, filename=output_filename, media_type="image/svg+xml")
    except Exception as e:
        cleanup_files(input_path)
        if os.path.exists(output_path):
            cleanup_files(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/svg/from-svg")
async def convert_from_svg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_format: str = Form("PNG"),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    quality: int = Form(95)
):
    """
    SVG转其他格式
    支持: PNG, JPG, PDF, EPS
    """
    if not file.filename.lower().endswith('.svg'):
        raise HTTPException(status_code=400, detail="文件必须是SVG格式")
    
    input_path = await save_upload_file(file)
    target_format = target_format.upper()
    
    # 确定输出文件扩展名
    ext_map = {
        "PNG": ".png",
        "JPG": ".jpg",
        "JPEG": ".jpg",
        "PDF": ".pdf",
        "EPS": ".eps",
        "PS": ".eps"
    }
    
    output_ext = ext_map.get(target_format, ".png")
    original_name = os.path.splitext(file.filename)[0]
    output_filename = f"{original_name}{output_ext}"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        if target_format == "PNG":
            SVGService.svg_to_png(input_path, output_path, width, height)
        elif target_format in ["JPG", "JPEG"]:
            SVGService.svg_to_jpg(input_path, output_path, width, height, quality)
        elif target_format == "PDF":
            SVGService.svg_to_pdf(input_path, output_path)
        elif target_format in ["EPS", "PS"]:
            SVGService.svg_to_eps(input_path, output_path)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的目标格式: {target_format}")
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(output_path, filename=output_filename)
    except Exception as e:
        cleanup_files(input_path)
        if os.path.exists(output_path):
            cleanup_files(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/svg/optimize")
async def optimize_svg(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    precision: int = Form(2),
    remove_metadata: bool = Form(True),
    remove_comments: bool = Form(True)
):
    """
    优化SVG文件
    压缩文件大小，不降低质量
    """
    if not file.filename.lower().endswith('.svg'):
        raise HTTPException(status_code=400, detail="文件必须是SVG格式")
    
    input_path = await save_upload_file(file)
    output_filename = f"optimized_{file.filename}"
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        SVGService.optimize_svg(
            input_path, 
            output_path, 
            precision=precision,
            remove_metadata=remove_metadata,
            remove_comments=remove_comments
        )
        
        # 获取优化前后的文件大小
        original_size = os.path.getsize(input_path)
        optimized_size = os.path.getsize(output_path)
        reduction = round((1 - optimized_size / original_size) * 100, 2)
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        
        response = FileResponse(output_path, filename=output_filename, media_type="image/svg+xml")
        response.headers["X-Original-Size"] = str(original_size)
        response.headers["X-Optimized-Size"] = str(optimized_size)
        response.headers["X-Size-Reduction"] = f"{reduction}%"
        
        return response
    except Exception as e:
        cleanup_files(input_path)
        if os.path.exists(output_path):
            cleanup_files(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/svg/info")
async def get_svg_info(file: UploadFile = File(...)):
    """
    获取SVG文件信息
    """
    if not file.filename.lower().endswith('.svg'):
        raise HTTPException(status_code=400, detail="文件必须是SVG格式")
    
    input_path = await save_upload_file(file)
    
    try:
        info = SVGService.get_svg_info(input_path)
        cleanup_files(input_path)
        return info
    except Exception as e:
        cleanup_files(input_path)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================
# 证件照生成端点
# ==============================================

@router.post("/image/id-photo")
async def generate_id_photo(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    size: str = Form(None),
    custom_width_mm: Optional[int] = Form(None),
    custom_height_mm: Optional[int] = Form(None),
    bg_color: str = Form("#FFFFFF"),
    beautify: bool = Form(True)
):
    """
    生成证件照
    - 自动抠图
    - 替换背景色
    - 可选美颜
    - 调整到标准尺寸或自定义尺寸
    
    参数：
    - size: 预设规格（如 1inch, 2inch）
    - custom_width_mm: 自定义宽度（毫米）
    - custom_height_mm: 自定义高度（毫米）
    """
    # 验证文件类型
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")
    
    # 验证尺寸参数
    if not size and not (custom_width_mm and custom_height_mm):
        raise HTTPException(status_code=400, detail="必须提供 size 或 custom_width_mm/custom_height_mm")
    
    input_path = await save_upload_file(file)
    
    if custom_width_mm and custom_height_mm:
        output_filename = f"id_photo_custom_{custom_width_mm}x{custom_height_mm}_{file.filename}"
    else:
        output_filename = f"id_photo_{size}_{file.filename}"
    
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        IdPhotoService.generate_id_photo(
            input_path=input_path,
            output_path=output_path,
            size_name=size,
            custom_width_mm=custom_width_mm,
            custom_height_mm=custom_height_mm,
            bg_color=bg_color,
            use_beautify=beautify
        )
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(
            output_path, 
            filename=output_filename,
            media_type="image/png"
        )
    except Exception as e:
        cleanup_files(input_path)
        if os.path.exists(output_path):
            cleanup_files(output_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/id-photo/render")
async def render_id_photo(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    crop_x: float = Form(...),
    crop_y: float = Form(...),
    crop_w: float = Form(...),
    crop_h: float = Form(...),
    target_w: int = Form(...),
    target_h: int = Form(...),
    rotate: float = Form(0),
    scale_x: float = Form(1),
    scale_y: float = Form(1),
    bg_color: str = Form(None),
    dpi: int = Form(300)
):
    """
    后端渲染证件照（高保真）
    """
    input_path = await save_upload_file(file)
    
    # 确定输出文件名
    original_name = os.path.splitext(file.filename)[0]
    output_filename = f"id_photo_render_{original_name}.png"
    
    output_path = str(settings.TEMP_DIR / f"{uuid.uuid4()}_{output_filename}")
    
    try:
        IdPhotoService.render_id_photo(
            input_path=input_path,
            output_path=output_path,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            target_w=target_w,
            target_h=target_h,
            rotate=rotate,
            scale_x=scale_x,
            scale_y=scale_y,
            bg_color=bg_color,
            dpi=dpi
        )
        
        background_tasks.add_task(cleanup_files, input_path, output_path)
        return FileResponse(
            output_path, 
            filename=output_filename,
            media_type="image/png"
        )
    except Exception as e:
        cleanup_files(input_path)
        if os.path.exists(output_path):
            cleanup_files(output_path)
        raise HTTPException(status_code=500, detail=str(e))

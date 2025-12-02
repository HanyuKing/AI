import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from server.core.config import settings
from server.services.file_service import FileService
from server.services.vision_service import VisionService
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

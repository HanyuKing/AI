import os
import io
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image

class FileTool:
    """文件处理工具类，提供PDF和图片的压缩、转换功能"""
    @staticmethod
    def compress_pdf(input_path: str, output_path: str, ratio: float) -> None:
        """
        压缩PDF文件 (通过缩小图片分辨率和JPEG压缩，保持矢量文字不变)
        
        Args:
            input_path: 输入PDF文件路径
            output_path: 输出PDF文件路径
            ratio: 期望的文件大小压缩比例 (0.0 < ratio <= 1.0)。
                   此实现会遍历PDF中的所有图片，将其长宽缩放到 sqrt(ratio)，
                   并使用JPEG压缩。矢量文字和图形保持不变。
        """
        if not (0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0 and 1")

        doc = fitz.open(input_path)
        
        # 分辨率缩放因子
        # 面积 ~ scale^2。为了达到目标大小比例，scale ~ sqrt(ratio)
        scale = ratio ** 0.5
        
        # 图片的JPEG压缩质量
        jpg_quality = 75
        
        # 跟踪已处理的图片，以正确处理复用的XObjects
        processed_xrefs = set()
        
        try:
            for page in doc:
                # get_images 返回列表 (xref, smask, width, height, bpc, colorspace, ...)
                for img in page.get_images():
                    xref = img[0]
                    if xref in processed_xrefs:
                        continue
                    processed_xrefs.add(xref)
                    
                    try:
                        # 获取图片内容作为Pixmap
                        # fitz.Pixmap(doc, xref) 提供原始图片数据（忽略smask）
                        pix = fitz.Pixmap(doc, xref)
                        
                        # 跳过小图片（图标等）或mask（bpc=1通常是mask/stencil）
                        # 注意：有些扫描文档bpc=1。我们只关注通常较大且为RGB/Gray的“照片”。
                        if pix.width < 100 or pix.height < 100:
                            continue
                            
                        # 验证每组件位数。如果<8，可能不是我们想要压缩的照片（例如1位文本掩码）。
                        # Pixmap.n 是每像素组件数。
                        
                        # 计算新尺寸
                        new_w = int(pix.width * scale)
                        new_h = int(pix.height * scale)
                        
                        if new_w < 1 or new_h < 1:
                            continue
                            
                        # 转换为PIL Image以进行高质量缩放
                        # pix.tobytes("png") 自动处理颜色转换为RGB/Gray（如果需要）
                        img_data = pix.tobytes("png")
                        
                        with Image.open(io.BytesIO(img_data)) as pil_img:
                            # 缩放
                            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            
                            output_buffer = io.BytesIO()
                            is_jpeg = False
                            
                            # 决定格式：
                            # 如果原始可能是照片（RGB/Gray），使用JPEG。
                            # 如果有透明度（从原始xref不太可能有，除非显式存储）或者是调色板，保持PNG？
                            # 注意：fitz.Pixmap(doc, xref) 通常没有alpha，除非这样存储。
                            
                            # 策略：
                            # 1. 如果是CMYK/RGB -> 转为RGB -> JPEG
                            # 2. 如果是Gray -> JPEG
                            # 3. 如果是RGBA（在原始xref中很少见）-> PNG
                            
                            if pil_img.mode in ('RGBA', 'LA'):
                                pil_img.save(output_buffer, format="PNG", optimize=True)
                            else:
                                # 如果不是Gray/RGB（例如CMYK, P），转为RGB
                                if pil_img.mode != 'L' and pil_img.mode != 'RGB':
                                    pil_img = pil_img.convert('RGB')
                                
                                pil_img.save(output_buffer, format="JPEG", quality=jpg_quality)
                                is_jpeg = True
                            
                            new_data = output_buffer.getvalue()
                        
                        # 更新PDF对象流
                        # compress=False 是关键：
                        # 1. 对于JPEG，我们不希望Deflate（它已经被压缩了）。
                        # 2. 如果我们Deflate一个JPEG但设置Filter=/DCTDecode，查看器会失败（期望原始JPEG）。
                        doc.update_stream(xref, new_data, compress=False)
                        
                        # 更新属性
                        doc.xref_set_key(xref, "Width", str(new_w))
                        doc.xref_set_key(xref, "Height", str(new_h))
                        
                        if is_jpeg:
                            doc.xref_set_key(xref, "Filter", "/DCTDecode")
                            doc.xref_set_key(xref, "BitsPerComponent", "8") # JPEG总是8位
                            
                            # 关键：如果存在DecodeParms则移除，因为JPEG (DCTDecode) 不支持它。
                            # 如果从之前的FlateDecode遗留下来，会损坏图片。
                            doc.xref_set_key(xref, "DecodeParms", "null")
                            
                            if pil_img.mode == 'L':
                                doc.xref_set_key(xref, "ColorSpace", "/DeviceGray")
                            else:
                                doc.xref_set_key(xref, "ColorSpace", "/DeviceRGB")
                        else:
                            doc.xref_set_key(xref, "Filter", "/FlateDecode")
                            # 对于PNG，可能需要小心ColorSpace，
                            # 但通常如果模式匹配，update_stream + Width/Height就足够了。
                            
                    except Exception as e:
                        # 如果一张图片失败，记录日志并继续
                        print(f"Warning: Failed to compress image xref {xref}: {e}")
                        continue

            doc.save(output_path, garbage=4, deflate=True)
        finally:
            doc.close()

    @staticmethod
    def compress_image_by_ratio(input_path: str, output_path: str, ratio: float) -> None:
        """
        按比例压缩图片 (目标是减小文件大小)

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            ratio: 期望的文件大小压缩比例 (0.0 < ratio <= 1.0)。
                   例如 ratio=0.5，目标是让文件大小变为原来的 50%。
                   (通过将长宽缩放到原来的 sqrt(ratio) 来近似实现)
        """
        if not (0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0 and 1")

        with Image.open(input_path) as img:
            width, height = img.size
            # Scale^2 = ratio => Scale = sqrt(ratio)
            scale = ratio ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 使用LANCZOS重采样进行高质量缩放
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 保存图片，保持原格式（如果可能）
            # 对于JPEG，默认quality=75
            resized_img.save(output_path)

    @staticmethod
    def compress_image_to_size(input_path: str, output_path: str, max_size_kb: int) -> None:
        """
        压缩图片到指定大小以内

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            max_size_kb: 最大文件大小 (KB)
        """
        max_size_bytes = max_size_kb * 1024
        
        # 检查原图大小
        if os.path.getsize(input_path) <= max_size_bytes:
            # 如果已经满足要求，直接复制或另存
            with Image.open(input_path) as img:
                img.save(output_path)
            return

        with Image.open(input_path) as img:
            # 如果是RGBA，转换为RGB以支持JPEG保存
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            # 二分法查找合适的质量参数
            min_quality = 5
            max_quality = 95
            target_quality = 95
            
            # 临时保存路径
            temp_path = output_path + ".tmp"
            
            while min_quality <= max_quality:
                quality = (min_quality + max_quality) // 2
                img.save(temp_path, "JPEG", quality=quality)
                
                size = os.path.getsize(temp_path)
                
                if size <= max_size_bytes:
                    target_quality = quality
                    min_quality = quality + 1
                else:
                    max_quality = quality - 1
            
            # 使用找到的最佳质量保存
            img.save(output_path, "JPEG", quality=target_quality)
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # 如果即使最低质量也无法满足，保留最低质量的结果，但不改变尺寸
            if os.path.getsize(output_path) > max_size_bytes:
                print(f"Warning: Could not compress to {max_size_kb}KB even with quality={min_quality}. Current size: {os.path.getsize(output_path)/1024:.2f}KB")


    @staticmethod
    def convert_image_format(input_path: str, output_path: str, target_format: Optional[str] = None) -> None:
        """
        转换图片格式，支持 PDF 和图片之间的相互转换。
        
        Args:
            input_path: 输入文件路径 (图片或PDF)
            output_path: 输出文件路径 (图片或PDF)
            target_format: 目标格式 (例如 "PNG", "JPEG", "PDF")。如果为None，则根据output_path后缀推断。
        """
        input_ext = os.path.splitext(input_path)[1].lower()
        output_ext = os.path.splitext(output_path)[1].lower()
        
        # 确定目标格式
        if target_format is None:
            if output_ext == '.pdf':
                target_format = 'PDF'
            elif output_ext in ['.jpg', '.jpeg']:
                target_format = 'JPEG'
            elif output_ext == '.png':
                target_format = 'PNG'
            elif output_ext == '.webp':
                target_format = 'WEBP'
            elif output_ext == '.bmp':
                target_format = 'BMP'
            elif output_ext == '.gif':
                target_format = 'GIF'
            elif output_ext == '.tiff':
                target_format = 'TIFF'
        
        if target_format:
            target_format = target_format.upper()

        # Case 1: PDF -> Image (Stitch all pages into one long image)
        if input_ext == '.pdf' and target_format != 'PDF':
            doc = fitz.open(input_path)
            try:
                if len(doc) > 0:
                    images = []
                    total_height = 0
                    max_width = 0
                    
                    # 1. Render all pages to images
                    for page in doc:
                        pix = page.get_pixmap()
                        
                        # Handle Alpha for JPEG
                        if target_format == 'JPEG' and pix.alpha:
                             pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Handle Alpha for formats that don't support it
                        no_alpha_formats = ['JPEG', 'BMP', 'PCX', 'PPM']
                        if (target_format in no_alpha_formats) and (img.mode in ['RGBA', 'LA', 'P']):
                             if img.mode == 'P':
                                img = img.convert('RGBA')
                             if img.mode in ['RGBA', 'LA']:
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                background.paste(img, mask=img.split()[-1])
                                img = background
                             else:
                                img = img.convert('RGB')
                        
                        images.append(img)
                        total_height += img.height
                        max_width = max(max_width, img.width)
                    
                    # 2. Stitch images
                    if not images:
                        raise ValueError("No images extracted from PDF")
                        
                    # Create blank canvas
                    # Use RGB for JPEG, RGBA for PNG (if supported/needed), but here we follow target_format logic roughly
                    # Simpler to just use RGB if we converted segments to RGB, or RGBA if we kept them.
                    # Since we normalized segments above, let's check the first image mode.
                    mode = images[0].mode
                    stitched_img = Image.new(mode, (max_width, total_height), (255, 255, 255) if mode == 'RGB' else (0, 0, 0, 0))
                    
                    y_offset = 0
                    for img in images:
                        # Center the image if it's narrower than max_width? Or left align?
                        # Usually left align or center. Let's left align for simplicity, or center.
                        # Let's left align to match document flow usually.
                        stitched_img.paste(img, (0, y_offset))
                        y_offset += img.height
                    
                    # 3. Save
                    stitched_img.save(output_path, format=target_format)
                    
                else:
                    raise ValueError("PDF file is empty")
            finally:
                doc.close()
            return

        # Case 2: Image -> PDF or Image -> Image
        with Image.open(input_path) as img:
            # Image -> PDF
            if target_format == 'PDF':
                # PDF不支持Alpha通道直接保存（通常），且需要RGB
                if img.mode == 'RGBA':
                    # 创建白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(output_path, "PDF", resolution=100.0)
                return

            # Image -> Image (Existing Logic)
            no_alpha_formats = ['JPEG', 'BMP', 'PCX', 'PPM']
            
            if (target_format in no_alpha_formats) and (img.mode in ['RGBA', 'LA', 'P']):
                if img.mode == 'P':
                    img = img.convert('RGBA')
                
                if img.mode in ['RGBA', 'LA']:
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            
            img.save(output_path, format=target_format)

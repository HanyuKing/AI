import os
import io
import asyncio
from typing import Optional, Callable, Awaitable
import fitz  # PyMuPDF
from PIL import Image

class FileTool:
    """文件处理工具类，提供PDF和图片的压缩、转换功能"""
    @staticmethod
    async def compress_pdf(
        input_path: str, 
        output_path: str, 
        ratio: float, 
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> None:
        """
        压缩PDF文件 (通过缩小图片分辨率和JPEG压缩，保持矢量文字不变)
        
        Args:
            input_path: 输入PDF文件路径
            output_path: 输出PDF文件路径
            ratio: 期望的文件大小压缩比例 (0.0 < ratio <= 1.0)。
            progress_callback: 异步回调函数，接收(progress_percent, message)
        """
        if not (0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0 and 1")

        if progress_callback:
            await progress_callback(0, "正在分析PDF文件...")

        # 在单独的线程中打开文档，避免阻塞主事件循环
        # fitz的操作是同步的，且可能很慢
        loop = asyncio.get_event_loop()
        
        def process_pdf_sync():
            doc = fitz.open(input_path)
            try:
                # 分辨率缩放因子
                scale = ratio ** 0.5
                jpg_quality = 75
                processed_xrefs = set()
                
                total_pages = len(doc)
                
                # 获取所有图片信息以计算总任务量
                # 为了简化进度，我们按页数来估算
                
                for page_num, page in enumerate(doc):
                    # 进度更新
                    # 由于是同步函数，我们不能直接await callback。
                    # 但我们可以使用 run_coroutine_threadsafe 回调到主循环
                    # 这里为了简单，我们先收集需要处理的图片，然后处理
                    pass

                # 重新遍历处理
                for page_num, page in enumerate(doc):
                    # 发送进度：页面级
                    if progress_callback:
                        current_percent = int((page_num / total_pages) * 90)
                        asyncio.run_coroutine_threadsafe(
                            progress_callback(current_percent, f"正在处理第 {page_num + 1}/{total_pages} 页..."),
                            loop
                        )

                    for img in page.get_images():
                        xref = img[0]
                        if xref in processed_xrefs:
                            continue
                        processed_xrefs.add(xref)
                        
                        try:
                            pix = fitz.Pixmap(doc, xref)
                            
                            if pix.width < 100 or pix.height < 100:
                                continue
                                
                            new_w = int(pix.width * scale)
                            new_h = int(pix.height * scale)
                            
                            if new_w < 1 or new_h < 1:
                                continue
                                
                            img_data = pix.tobytes("png")
                            
                            with Image.open(io.BytesIO(img_data)) as pil_img:
                                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                
                                output_buffer = io.BytesIO()
                                is_jpeg = False
                                
                                if pil_img.mode in ('RGBA', 'LA'):
                                    pil_img.save(output_buffer, format="PNG", optimize=True)
                                else:
                                    if pil_img.mode != 'L' and pil_img.mode != 'RGB':
                                        pil_img = pil_img.convert('RGB')
                                    pil_img.save(output_buffer, format="JPEG", quality=jpg_quality)
                                    is_jpeg = True
                                
                                new_data = output_buffer.getvalue()
                            
                            doc.update_stream(xref, new_data, compress=False)
                            doc.xref_set_key(xref, "Width", str(new_w))
                            doc.xref_set_key(xref, "Height", str(new_h))
                            
                            if is_jpeg:
                                doc.xref_set_key(xref, "Filter", "/DCTDecode")
                                doc.xref_set_key(xref, "BitsPerComponent", "8")
                                doc.xref_set_key(xref, "DecodeParms", "null")
                                if pil_img.mode == 'L':
                                    doc.xref_set_key(xref, "ColorSpace", "/DeviceGray")
                                else:
                                    doc.xref_set_key(xref, "ColorSpace", "/DeviceRGB")
                            else:
                                doc.xref_set_key(xref, "Filter", "/FlateDecode")
                                
                        except Exception as e:
                            print(f"Warning: Failed to compress image xref {xref}: {e}")
                            continue

                if progress_callback:
                     asyncio.run_coroutine_threadsafe(
                        progress_callback(95, "正在保存文件..."),
                        loop
                    )
                    
                doc.save(output_path, garbage=4, deflate=True)
                
            finally:
                doc.close()

        # 在线程池中运行同步的PDF处理
        await loop.run_in_executor(None, process_pdf_sync)
        
        if progress_callback:
            await progress_callback(100, "处理完成")

    @staticmethod
    def convert_image_format(input_path: str, output_path: str, target_format: Optional[str] = None) -> None:
        # (Same content as before)
        input_ext = os.path.splitext(input_path)[1].lower()
        output_ext = os.path.splitext(output_path)[1].lower()
        
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

        if input_ext == '.pdf' and target_format != 'PDF':
            doc = fitz.open(input_path)
            try:
                if len(doc) > 0:
                    images = []
                    total_height = 0
                    max_width = 0
                    
                    for page in doc:
                        pix = page.get_pixmap()
                        
                        if target_format == 'JPEG' and pix.alpha:
                             pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
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
                    
                    if not images:
                        raise ValueError("No images extracted from PDF")
                        
                    mode = images[0].mode
                    stitched_img = Image.new(mode, (max_width, total_height), (255, 255, 255) if mode == 'RGB' else (0, 0, 0, 0))
                    
                    y_offset = 0
                    for img in images:
                        stitched_img.paste(img, (0, y_offset))
                        y_offset += img.height
                    
                    stitched_img.save(output_path, format=target_format)
                    
                else:
                    raise ValueError("PDF file is empty")
            finally:
                doc.close()
            return

        with Image.open(input_path) as img:
            if target_format == 'PDF':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(output_path, "PDF", resolution=100.0)
                return

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

    @staticmethod
    def compress_image_by_ratio(input_path: str, output_path: str, ratio: float) -> None:
        # (Same as before)
        if not (0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0 and 1")

        with Image.open(input_path) as img:
            width, height = img.size
            scale = ratio ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path)

    @staticmethod
    def compress_image_to_size(input_path: str, output_path: str, max_size_kb: int) -> None:
        # (Same as before)
        max_size_bytes = max_size_kb * 1024
        
        if os.path.getsize(input_path) <= max_size_bytes:
            with Image.open(input_path) as img:
                img.save(output_path)
            return

        with Image.open(input_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            min_quality = 5
            max_quality = 95
            target_quality = 95
            
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
            
            img.save(output_path, "JPEG", quality=target_quality)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if os.path.getsize(output_path) > max_size_bytes:
                print(f"Warning: Could not compress to {max_size_kb}KB even with quality={min_quality}.")

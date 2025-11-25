import os
from typing import Optional
import fitz  # PyMuPDF

from PIL import Image

class FileTool:
    """文件处理工具类，提供PDF和图片的压缩、转换功能"""
    @staticmethod
    def compress_pdf(input_path: str, output_path: str, ratio: float) -> None:
        """
        压缩PDF文件 (通过将页面转换为图片重组PDF)
        
        Args:
            input_path: 输入PDF文件路径
            output_path: 输出PDF文件路径
            ratio: 期望的文件大小压缩比例 (0.0 < ratio <= 1.0)。
                   此实现会将页面光栅化为图片，通过降低分辨率和JPEG压缩来减小体积。
        """
        if not (0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0 and 1")

        doc = fitz.open(input_path)
        out_doc = fitz.open()
        
        # Scale factor for resolution
        # Area ~ scale^2. To get target size ratio, scale ~ sqrt(ratio)
        scale = ratio ** 0.5
        
        # JPEG quality
        jpg_quality = 75
        
        try:
            for page in doc:
                # Render page to image
                # matrix controls the resolution
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                
                # Create new page with original dimensions
                new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
                
                # Insert the rendered image into the new page
                # stream=pix.tobytes("jpg") ensures we use JPEG compression
                new_page.insert_image(page.rect, stream=pix.tobytes("jpg", jpg_quality=jpg_quality))
                
            out_doc.save(output_path, garbage=4, deflate=True)
        finally:
            doc.close()
            out_doc.close()

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
        转换图片格式

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            target_format: 目标格式 (例如 "PNG", "JPEG")。如果为None，则根据output_path后缀推断。
        """
        with Image.open(input_path) as img:
            # 确定目标格式
            if target_format is None:
                # 从后缀推断
                ext = os.path.splitext(output_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    target_format = 'JPEG'
                elif ext == '.png':
                    target_format = 'PNG'
                elif ext == '.webp':
                    target_format = 'WEBP'
                elif ext == '.bmp':
                    target_format = 'BMP'
                elif ext == '.gif':
                    target_format = 'GIF'
                elif ext == '.tiff':
                    target_format = 'TIFF'
            
            if target_format:
                target_format = target_format.upper()
            
            # 需要移除Alpha通道的格式列表
            no_alpha_formats = ['JPEG', 'BMP', 'PCX', 'PPM']
            
            # 如果目标格式不支持Alpha，或者强制转为RGB
            if (target_format in no_alpha_formats) and (img.mode in ['RGBA', 'LA', 'P']):
                # 如果是P模式（调色板），先转RGBA
                if img.mode == 'P':
                    img = img.convert('RGBA')
                
                if img.mode in ['RGBA', 'LA']:
                    # 创建白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    # 使用alpha通道作为mask进行粘贴
                    # split()[-1] 获取最后一个通道，即Alpha
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            
            img.save(output_path, format=target_format)

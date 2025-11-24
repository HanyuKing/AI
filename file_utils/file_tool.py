import os
from typing import Optional
import fitz  # PyMuPDF

from PIL import Image

class FileTool:
    """文件处理工具类，提供PDF和图片的压缩、转换功能"""
    @staticmethod
    def compress_pdf(input_path: str, output_path: str, ratio: float) -> None:
        """
        压缩PDF文件 (通过压缩内部图片来减小文件大小)

        Args:
            input_path: 输入PDF文件路径
            output_path: 输出PDF文件路径
            ratio: 期望的文件大小压缩比例 (0.0 < ratio <= 1.0)。
                   例如 ratio=0.8，目标是让文件大小变为原来的 80%。
                   (通过将图片像素总量缩放到原来的 ratio 来近似实现)
        """
        if not (0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0 and 1")

        doc = fitz.open(input_path)
        
        # 计算缩放因子 (dimensions scale factor)
        # Area_new = Area_old * ratio
        # Width_new * Height_new = Width_old * Height_old * ratio
        # Scale^2 = ratio => Scale = sqrt(ratio)
        scale_factor = ratio ** 0.5

        processed_xrefs = set()

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                smask = img_info[1]
                
                if xref in processed_xrefs:
                    continue
                processed_xrefs.add(xref)
                
                # 提取图片 (使用Pixmap以正确处理mask/alpha)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    
                    # 如果有mask，合并它
                    if smask > 0:
                        mask = fitz.Pixmap(doc, smask)
                        try:
                            pix = fitz.Pixmap(pix, mask)
                        except Exception as e:
                            print(f"Warning: Failed to combine mask for xref {xref}: {e}")
                    
                    # 如果是CMYK，先转RGB (Pillow处理CMYK可能有问题，且我们最终要存JPEG)
                    if pix.colorspace.n == 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    image_bytes = pix.tobytes("png") # 导出为PNG以保留Alpha
                except Exception as e:
                    print(f"Warning: Failed to extract image xref {xref}: {e}")
                    continue
                
                # 使用Pillow压缩
                try:
                    import io
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        width, height = img.size
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        
                        # 避免缩得太小
                        if new_width < 10 or new_height < 10:
                            continue
                            
                        # 缩放
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # 转为JPEG字节流
                        buffer = io.BytesIO()
                        
                        # 处理颜色模式
                        if img.mode == "RGBA":
                            # 创建白色背景
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            # 使用alpha通道作为mask进行粘贴
                            background.paste(img, mask=img.split()[3])
                            img = background
                        elif img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # 使用中等质量
                        img.save(buffer, format="JPEG", quality=75)
                        new_image_bytes = buffer.getvalue()
                        
                        # 使用 page.replace_image 更新图片 (会自动处理Width/Height等元数据)
                        # 注意：虽然是在 page 上调用，但如果是共享图片，会更新全局对象
                        page.replace_image(xref, stream=new_image_bytes)
                        
                except Exception as e:
                    print(f"Warning: Failed to compress image xref {xref}: {e}")
                    continue

        # 保存并执行垃圾回收/压缩
        doc.save(output_path, garbage=4, deflate=True)
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

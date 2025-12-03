import os
from typing import Tuple, Optional, Dict, List

# 规格数据可以立即导入（不依赖重量级库）
from practices.id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS, get_spec_by_id, search_specs

# 向后兼容：导入 ID_PHOTO_SIZES
# 延迟导入 IdPhotoProcessor 以避免启动时加载 rembg
def _lazy_import_processor():
    """延迟导入 IdPhotoProcessor，仅在实际使用时加载"""
    from practices.id_photo_utils import IdPhotoProcessor, ID_PHOTO_SIZES
    return IdPhotoProcessor, ID_PHOTO_SIZES

# 创建一个缓存的导入
_processor_class = None
_photo_sizes = None

class IdPhotoService:
    @staticmethod
    def generate_id_photo(
        input_path: str,
        output_path: str,
        size_name: str = "1inch",
        bg_color: str = "#FFFFFF",
        use_beautify: bool = True,
        spec_id: str = None  # 新API参数
    ) -> None:
        """
        Generate ID photo from input image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            size_name: [Deprecated] Size name (key in ID_PHOTO_SIZES), kept for backward compatibility
            bg_color: Hex color string (e.g. "#FFFFFF", "#438EDB", "transparent")
            use_beautify: Whether to apply beautification
            spec_id: [New] Spec ID (e.g. "driving_license", "passport", "civil_servant")
        """
        # 延迟导入 IdPhotoProcessor
        global _processor_class
        if _processor_class is None:
            IdPhotoProcessor, _ = _lazy_import_processor()
            _processor_class = IdPhotoProcessor
        else:
            IdPhotoProcessor = _processor_class
        
        # 优先使用新的 spec_id，如果没有则使用旧的 size_name
        if spec_id:
            final_spec_id = spec_id
        else:
            final_spec_id = size_name
        
        # Parse hex color to RGB tuple
        if bg_color.lower() == 'transparent':
            bg_rgb = None  # Indicate transparent background
        else:
            bg_color = bg_color.lstrip('#')
            if len(bg_color) == 6:
                r = int(bg_color[0:2], 16)
                g = int(bg_color[2:4], 16)
                b = int(bg_color[4:6], 16)
                bg_rgb = (r, g, b)
            else:
                bg_rgb = (255, 255, 255)  # Default white

        processor = IdPhotoProcessor(image_path=input_path)
        
        try:
            result_img = processor.generate_id_photo(
                spec_id=final_spec_id,
                bg_color=bg_rgb,
                use_beautify=use_beautify
            )
            
            # 自动选择保存格式
            if bg_rgb is None:
                # 透明背景保存为 PNG
                result_img.save(output_path, "PNG", dpi=(300, 300))
            else:
                # 有底色保存为 JPG
                if output_path.lower().endswith('.png'):
                    result_img.save(output_path, "PNG", dpi=(300, 300))
                else:
                    result_img.save(output_path, "JPEG", quality=95, dpi=(300, 300))
        except Exception as e:
            raise RuntimeError(f"Failed to generate ID photo: {str(e)}")

    @staticmethod
    def get_supported_sizes() -> dict:
        """获取支持的尺寸（旧API，向后兼容）"""
        global _photo_sizes
        if _photo_sizes is None:
            _, ID_PHOTO_SIZES = _lazy_import_processor()
            _photo_sizes = ID_PHOTO_SIZES
        return _photo_sizes
    
    @staticmethod
    def get_all_specs() -> Dict:
        """获取所有规格分类（新API）"""
        return ID_PHOTO_SPECS
    
    @staticmethod
    def get_spec_info(spec_id: str) -> Optional[Dict]:
        """获取指定规格的详细信息（新API）"""
        return get_spec_by_id(spec_id)
    
    @staticmethod
    def search_specs(keyword: str) -> List[Dict]:
        """搜索规格（新API）"""
        return search_specs(keyword)
    
    @staticmethod
    def get_background_colors() -> Dict:
        """获取背景颜色预设（新API）"""
        return BG_COLOR_PRESETS



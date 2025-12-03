import os
from typing import Tuple, Optional
from practices.id_photo_utils import IdPhotoProcessor, ID_PHOTO_SIZES

class IdPhotoService:
    @staticmethod
    def generate_id_photo(
        input_path: str,
        output_path: str,
        size_name: str = "1inch",
        bg_color: str = "#FFFFFF",
        use_beautify: bool = True
    ) -> None:
        """
        Generate ID photo from input image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            size_name: Size name (key in ID_PHOTO_SIZES)
            bg_color: Hex color string (e.g. "#FFFFFF", "#438EDB")
            use_beautify: Whether to apply beautification
        """
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
                bg_rgb = (255, 255, 255) # Default white

        processor = IdPhotoProcessor(image_path=input_path)
        
        try:
            result_img = processor.generate_id_photo(
                size_name=size_name,
                bg_color=bg_rgb,
                use_beautify=use_beautify
            )
            result_img.save(output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to generate ID photo: {str(e)}")

    @staticmethod
    def get_supported_sizes() -> dict:
        return ID_PHOTO_SIZES



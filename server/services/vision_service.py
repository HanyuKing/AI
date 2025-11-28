import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from image_utils.image_transform_util import rotate_image_numpy, resize_image

class VisionService:
    @staticmethod
    def rotate_image(input_path: str, output_path: str, angle: float) -> None:
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Could not read image")
            
        # Rotate
        # Note: The utility function expects degrees. Positive is counter-clockwise.
        # We keep consistency with that.
        rotated = rotate_image_numpy(img, angle)
        
        # Save
        cv2.imwrite(output_path, rotated)

    @staticmethod
    def resize_image(input_path: str, output_path: str, width: int = None, height: int = None) -> None:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Could not read image")
            
        resized = resize_image(img, width=width, height=height)
        cv2.imwrite(output_path, resized)


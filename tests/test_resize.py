import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_utils.image_transform_util import resize_image

def test_resize():
    # Create a dummy image (100x200)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    
    print("Original shape:", img.shape)

    # Test 1: Resize with specific width and height
    resized1 = resize_image(img, width=50, height=50)
    print("Test 1 (50x50):", resized1.shape)
    assert resized1.shape == (50, 50, 3)

    # Test 2: Resize with width only (maintain aspect ratio)
    # Original 200w x 100h. Target width 100. Ratio 0.5. Target height should be 50.
    resized2 = resize_image(img, width=100)
    print("Test 2 (width=100):", resized2.shape)
    assert resized2.shape == (50, 100, 3)

    # Test 3: Resize with height only (maintain aspect ratio)
    # Original 200w x 100h. Target height 200. Ratio 2.0. Target width should be 400.
    resized3 = resize_image(img, height=200)
    print("Test 3 (height=200):", resized3.shape)
    assert resized3.shape == (200, 400, 3)

    # Test 4: No resize
    resized4 = resize_image(img)
    print("Test 4 (no resize):", resized4.shape)
    assert resized4.shape == (100, 200, 3)

    print("All tests passed!")

if __name__ == "__main__":
    test_resize()

import cv2
import numpy as np
from PIL import Image

def rgb_to_cmyk(rgb):
    """rgb: uint8 numpy array (H, W, 3)，返回 cmyk uint8 数组 (H, W, 4)"""
    rgb = rgb.astype(np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    K = 1 - np.max(rgb, axis=2)

    denom = (1 - K)
    denom[denom == 0] = 1

    C = (1 - R - K) / denom
    M = (1 - G - K) / denom
    Y = (1 - B - K) / denom

    cmyk = np.stack([C, M, Y, K], axis=-1)
    return (cmyk * 255).astype(np.uint8)


def save_cmyk_tiff(cmyk, filename="output_cmyk.tif"):
    """
    将 CMYK 四通道保存为真正的 CMYK TIFF。
    Pillow 支持 CMYK 模式并会写入 4 个通道。
    """
    img_cmyk = Image.fromarray(cmyk, mode="CMYK")
    img_cmyk.save(filename, compression="tiff_deflate")
    print("✅ 已保存 CMYK 四通道到 TIFF:", filename)


if __name__ == "__main__":
    # 读取 RGB
    rgb = cv2.imread("input/raw.png")
    if rgb is None:
        raise FileNotFoundError("❌ 无法读取 input.jpg")

    # BGR → RGB
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # RGB → CMYK
    cmyk = rgb_to_cmyk(rgb)

    # 保存为 CMYK TIFF（四通道）
    save_cmyk_tiff(cmyk, "output/image_cmyk.tif")

    img = Image.open("output/image_cmyk.tif")
    print(img.mode)

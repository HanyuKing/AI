import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
import io

class IdPhotoProcessor:
    def __init__(self, image_path: str = None, image_data: bytes = None):
        """
        初始化，支持传入文件路径或原始图像字节数据。
        """
        if image_path:
            self.original_image = Image.open(image_path)
        elif image_data:
            self.original_image = Image.open(io.BytesIO(image_data))
        else:
            raise ValueError("必须提供 image_path 或 image_data")
        
        # 确保图像是 RGB 模式
        if self.original_image.mode != 'RGB':
            self.original_image = self.original_image.convert('RGB')
            
        # 初始化 rembg 会话，使用 SOTA 模型 BiRefNet-portrait
        # 该模型对人像边缘（尤其是头发）的处理远优于 u2net
        # 注意：首次运行会自动下载约 1GB 的模型文件
        try:
            self.session = new_session("birefnet-portrait")
        except Exception as e:
            print(f"警告: 无法加载 birefnet-portrait 模型，尝试回退到 u2net_human_seg。错误: {e}")
            try:
                self.session = new_session("u2net_human_seg")
            except Exception:
                self.session = new_session("u2net")

    def _post_process(self, image: Image.Image, remove_stray_hairs: bool = False) -> Image.Image:
        """
        对抠图结果进行后处理：
        1. 腐蚀边缘以去除白边/杂色。
        2. (可选) 形态学开运算去除杂毛。
        3. Alpha 通道 Gamma 校正，收紧边缘。
        """
        # 转换为 numpy 数组
        img_np = np.array(image)
        
        # 获取 Alpha 通道
        alpha = img_np[:, :, 3]
        
        # 1. 腐蚀操作：收缩 Alpha 通道
        # BiRefNet 精度很高，通常不需要太强的腐蚀，这里仅做微调
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        
        # 2. 去除杂毛（形态学开运算：先腐蚀后膨胀）
        # 默认关闭，BiRefNet 通常能保留很好的发丝细节，不需要这个破坏性操作
        if remove_stray_hairs:
            clean_kernel = np.ones((5, 5), np.uint8)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, clean_kernel)
        
        # 3. 高斯模糊：柔化边缘
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        
        # 4. Gamma 校正：收紧 Alpha 通道
        # BiRefNet 的 alpha 通道通常已经很准了，这里做一点点收紧即可
        alpha_norm = alpha.astype(float) / 255.0
        alpha_norm = np.power(alpha_norm, 1.2)  # Gamma 调低一点 (1.5 -> 1.2)
        alpha = (alpha_norm * 255).astype(np.uint8)
        
        # 更新 Alpha 通道
        img_np[:, :, 3] = alpha
        
        return Image.fromarray(img_np)

    def remove_background(self, use_alpha_matting: bool = True, use_post_process: bool = True, 
                         erode_size: int = 10, remove_stray_hairs: bool = False) -> Image.Image:
        """
        使用 rembg 移除背景。
        use_alpha_matting: 是否使用 Alpha Matting 技术（推荐 True）。
        use_post_process: 是否进行边缘腐蚀后处理（推荐 True）。
        erode_size: Alpha Matting 的腐蚀大小。BiRefNet 精度高，默认值调小至 10。
        remove_stray_hairs: 是否尝试去除杂毛（默认 False）。
        返回一个 RGBA 模式的 PIL Image（背景透明）。
        """
        # 配置 alpha matting 参数
        # Alpha Matting 是一种边缘羽化技术，用于处理半透明区域（如头发、毛玻璃）
        kwargs = {
            'session': self.session
        }
        if use_alpha_matting:
            kwargs.update({
                'alpha_matting': True,
                
                # 前景阈值 (0-255)：
                'alpha_matting_foreground_threshold': 240,
                
                # 背景阈值 (0-255)：
                'alpha_matting_background_threshold': 10,
                
                # 腐蚀大小：
                # BiRefNet 边缘通常很准，不需要太大的腐蚀
                'alpha_matting_erode_size': erode_size, 
            })
            
        output = remove(self.original_image, **kwargs)
        
        if use_post_process:
            output = self._post_process(output, remove_stray_hairs=remove_stray_hairs)
            
        return output

    def add_background_color(self, bg_color: tuple = (255, 0, 0), use_alpha_matting: bool = True,
                            erode_size: int = 10, remove_stray_hairs: bool = False) -> Image.Image:
        """
        移除背景并替换为纯色背景。
        bg_color: (R, G, B) 元组，例如 (255, 0, 0) 代表红色。
        use_alpha_matting: 是否使用 Alpha Matting 优化边缘（推荐 True）。
        erode_size: Alpha Matting 腐蚀大小。
        remove_stray_hairs: 是否去除杂毛。
        """
        # 1. 获取背景透明的图像（默认启用后处理）
        fg = self.remove_background(use_alpha_matting=use_alpha_matting, 
                                  use_post_process=True, 
                                  erode_size=erode_size,
                                  remove_stray_hairs=remove_stray_hairs)

        fg.save("./output/rembg.png")
        
        # 2. 创建纯色背景
        bg = Image.new("RGB", fg.size, bg_color)
        
        # 3. 将前景合成到背景上
        # 使用前景的 alpha 通道作为掩码进行粘贴
        bg.paste(fg, (0, 0), fg)
        
        return bg

    def beautify(self, smooth_strength: int = 10, brighten_strength: float = 1.2) -> Image.Image:
        """
        应用磨皮和美白（提亮）。
        smooth_strength: 双边滤波强度（值越大越平滑）。
        brighten_strength: 亮度倍数（1.0 为原图亮度）。
        """
        # 将 PIL 转换为 OpenCV 格式 (RGB -> BGR)
        img_np = np.array(self.original_image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. 磨皮（使用双边滤波）
        # d: 像素邻域直径
        # sigmaColor: 颜色空间的标准差
        # sigmaSpace: 坐标空间的标准差
        # 人像磨皮通常取 10-20 左右的值
        smoothed = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=smooth_strength*5, sigmaSpace=smooth_strength*5)

        # 2. 美白（提亮）
        # 转换到 HSV 空间调整 V (亮度) 分量
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 应用亮度倍数并截断到 0-255
        v = cv2.multiply(v, brighten_strength)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        final_hsv = cv2.merge((h, s, v))
        brightened_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # 转回 PIL 格式 (BGR -> RGB)
        final_rgb = cv2.cvtColor(brightened_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)

if __name__ == "__main__":
    # 简单的命令行测试
    import sys
    import os
    img_name = "工卡照.jpeg"
    input_path = "images/" + img_name

    processor = IdPhotoProcessor(input_path)
    
    # 测试换底（红底）
    red_bg_img = processor.add_background_color((255, 0, 0))
    red_bg_img.save("output/change_bg_" + img_name)

    # 测试美颜
    beauty_img = processor.beautify()
    beauty_img.save("output/beauty_" + img_name)

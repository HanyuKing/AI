import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
import io

from server.utils.id_photo_specs import ID_PHOTO_SPECS, BG_COLOR_PRESETS, get_spec_by_id, search_specs

# 向后兼容：保留旧的 ID_PHOTO_SIZES
ID_PHOTO_SIZES = {
    "1inch": (25, 35),
    "small_1inch": (22, 32),
    "large_1inch": (33, 48),
    "2inch": (35, 49),
    "small_2inch": (33, 48),
    "large_2inch": (35, 53),
    "5inch": (89, 127),
}

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
        
        # 确保图像是 RGB 或 RGBA 模式
        if self.original_image.mode not in ('RGB', 'RGBA'):
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

    def generate_id_photo(
        self,
        spec_id: str = None,
        size_name: str = None,  # 向后兼容旧API
        bg_color: tuple = None,
        bg_color_name: str = None,
        use_beautify: bool = True,
        dpi: int = 300,
        custom_width_mm: int = None,  # 自定义宽度（毫米）
        custom_height_mm: int = None  # 自定义高度（毫米）
    ) -> Image.Image:
        """
        生成标准证件照。
        
        Args:
            spec_id: 规格ID（如 "passport", "driving_license", "civil_servant" 等）
            size_name: [已弃用] 旧的尺寸名称（如 "1inch", "2inch"），为了向后兼容保留
            bg_color: 背景颜色 (R, G, B)，如果是 None 且未指定 bg_color_name，则返回透明背景
            bg_color_name: 背景颜色名称（如 "white", "blue", "red"），会覆盖 bg_color
            use_beautify: 是否使用美颜功能
            dpi: 输出图片的 DPI（默认 300）
            custom_width_mm: 自定义宽度（毫米），优先级高于spec_id
            custom_height_mm: 自定义高度（毫米），优先级高于spec_id
            
        Returns:
            PIL Image 对象
        """
        # 1. 确定尺寸
        spec = None
        
        # 优先使用自定义尺寸
        if custom_width_mm and custom_height_mm:
            spec = {
                "name": "自定义",
                "size_mm": (custom_width_mm, custom_height_mm),
                "description": f"自定义尺寸 {custom_width_mm}×{custom_height_mm}mm"
            }
        else:
            # 向后兼容：如果使用了旧的 size_name 参数
            if size_name and not spec_id:
                spec_id = size_name
            elif not spec_id and not size_name:
                spec_id = "1inch"  # 默认值
            
            # 获取规格信息
            spec = get_spec_by_id(spec_id)
            
            # 如果新规格系统中找不到，尝试从旧的 ID_PHOTO_SIZES 中获取
            if spec is None and spec_id in ID_PHOTO_SIZES:
                # 创建一个临时的 spec 对象用于向后兼容
                width_mm, height_mm = ID_PHOTO_SIZES[spec_id]
                spec = {
                    "name": spec_id,
                    "size_mm": (width_mm, height_mm),
                    "description": f"标准 {spec_id} 尺寸"
                }
            elif spec is None:
                raise ValueError(f"不支持的规格ID: {spec_id}")
        
        # 2. 处理背景颜色
        if bg_color_name:
            if bg_color_name not in BG_COLOR_PRESETS:
                raise ValueError(f"不支持的背景颜色: {bg_color_name}")
            bg_color = BG_COLOR_PRESETS[bg_color_name]["rgb"]
        
        # 3. 可选：美颜处理（降低强度，避免眼镜等细节模糊）
        if use_beautify:
            # 降低平滑强度：从 5 降到 2，保留更多细节
            self.original_image = self.beautify(smooth_strength=2, brighten_strength=1.1)
        
        # 4. 抠图并换背景
        if bg_color is None:
            # 仅移除背景，返回透明 PNG
            result_img = self.remove_background(
                use_alpha_matting=True,
                use_post_process=True,
                erode_size=10,
                remove_stray_hairs=False
            )
        else:
            # 移除背景并添加底色
            result_img = self.add_background_color(
                bg_color=bg_color,
                use_alpha_matting=True,
                erode_size=10,
                remove_stray_hairs=False
            )
        
        # 5. 调整到标准尺寸
        width_mm, height_mm = spec["size_mm"]
        
        # 毫米转像素 (DPI = 300 时，1mm = 11.811 像素)
        mm_to_px = dpi / 25.4
        target_width_px = int(width_mm * mm_to_px)
        target_height_px = int(height_mm * mm_to_px)
        
        # 6. 智能裁剪/缩放到目标尺寸
        result_img = self._smart_crop_and_resize(result_img, target_width_px, target_height_px)
        
        # 7. 设置 DPI 元数据
        result_img.info['dpi'] = (dpi, dpi)
        
        return result_img

    def _smart_crop_and_resize(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """
        智能裁剪和缩放图片到目标尺寸。
        
        策略：
        1. 计算目标宽高比
        2. 如果原图宽高比不匹配，先按比例缩放，然后居中裁剪
        3. 最后精确缩放到目标尺寸
        """
        orig_width, orig_height = img.size
        target_ratio = target_width / target_height
        orig_ratio = orig_width / orig_height
        
        if abs(orig_ratio - target_ratio) < 0.01:
            # 宽高比已经接近，直接缩放
            return img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 需要裁剪
        if orig_ratio > target_ratio:
            # 原图更宽，以高度为基准，裁剪宽度
            new_height = orig_height
            new_width = int(new_height * target_ratio)
            left = (orig_width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = orig_height
        else:
            # 原图更高，以宽度为基准，裁剪高度（从上部裁剪，保留人脸）
            new_width = orig_width
            new_height = int(new_width / target_ratio)
            left = 0
            top = 0  # 从顶部开始，保留头部
            right = orig_width
            bottom = new_height
        
        # 裁剪并缩放
        cropped = img.crop((left, top, right, bottom))
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

    def render_id_photo(
        self,
        crop_x: float,
        crop_y: float,
        crop_w: float,
        crop_h: float,
        target_w: int,
        target_h: int,
        rotate: float = 0,
        scale_x: float = 1,
        scale_y: float = 1,
        bg_color: str = None,
        dpi: int = 300
    ) -> Image.Image:
        """
        根据前端裁剪参数，后端高保真渲染最终图片。
        使用 Lanczos 滤镜进行重采样，确保最佳清晰度。
        """
        img = self.original_image.copy()

        # 1. 旋转
        if rotate != 0:
            # expand=True 确保旋转后不被裁剪，保持完整画面
            # Pillow rotate 是逆时针，Cropper.js rotate 是顺时针，所以取负
            img = img.rotate(-rotate, expand=True, resample=Image.Resampling.BICUBIC)

        # 2. 翻转
        if scale_x == -1:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if scale_y == -1:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # 3. 裁剪
        # 确保坐标是整数
        x = int(round(crop_x))
        y = int(round(crop_y))
        w = int(round(crop_w))
        h = int(round(crop_h))
        
        # 边界检查
        img_w, img_h = img.size
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        img = img.crop((x, y, x + w, y + h))

        # 4. 缩放到目标尺寸 (核心步骤：使用 Lanczos)
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 5. 锐化 (针对眼镜等细节优化)
        try:
            from PIL import ImageEnhance
            # 适度锐化，增强边缘清晰度
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)  # 1.0 是原图，1.2 是轻微锐化
        except Exception as e:
            print(f"锐化失败: {e}")

        # 6. 添加背景色
        if bg_color and bg_color != 'transparent':
            try:
                from PIL import ImageColor
                # 解析颜色
                if bg_color.startswith('#'):
                    color = ImageColor.getrgb(bg_color)
                else:
                    # 尝试直接解析
                    color = ImageColor.getrgb(bg_color)
                
                bg = Image.new("RGB", img.size, color)
                # 使用 img 的 alpha 通道作为掩码
                if img.mode == 'RGBA':
                    bg.paste(img, (0, 0), img)
                    img = bg
                else:
                    bg.paste(img, (0, 0))
                    img = bg
            except Exception as e:
                print(f"背景色处理失败: {e}")
                # 忽略错误，返回透明背景

        # 7. 设置 DPI
        img.info['dpi'] = (dpi, dpi)

        return img

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

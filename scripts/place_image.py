#!/usr/bin/env python3
"""
stick_logo_on_cup_v2.py
改进版：更自然的“贴纸”效果，避免凹凸、变形和裁剪问题
"""

import cv2
import numpy as np

def cylindrical_warp(image, curve=0.3):
    """
    简单模拟圆柱形弯曲。
    curve > 0: 水平弯曲度 (0~1)，越大越弯。
    """
    h, w = image.shape[:2]
    radius = w / curve
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for y in range(h):
        for x in range(w):
            theta = (x - w/2) / radius
            map_x[y, x] = radius * np.sin(theta) + w/2
            map_y[y, x] = y
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def trim_transparent_border(image, alpha_threshold=5):
    """
    去除图像四周的全透明区域，只保留非透明的最小外接矩形。
    alpha_threshold: 大于该阈值视为非透明
    """
    if image.ndim < 3 or image.shape[2] < 4:
        return image
    alpha = image[:, :, 3]
    mask = alpha > alpha_threshold
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]

def place_logo(cup_img, logo_img, position="center", scale=0.3, alpha=1.0, curve=0.2):
    """
    把 logo 贴到杯子图上（自然贴纸效果）
    - position: "center" / (x, y)
    - scale: logo 相对于杯子宽度比例；None 表示自动
    - alpha: 透明度 0~1
    - curve: 圆柱弯曲程度
    """

    # 转 RGBA
    if logo_img.shape[2] == 3:
        logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2BGRA)
    # 先去除透明边界，再计算宽高
    logo_img = trim_transparent_border(logo_img, alpha_threshold=5)
    cup = cup_img.copy()

    h_cup, w_cup = cup.shape[:2]
    h_logo, w_logo = logo_img.shape[:2]

    # 自适应缩放：对宽高设置最小/最大占比，确保观感自然
    # 经验值：宽占比 15%~50%，高占比 8%~35%，默认目标宽占比 35%
    min_w_ratio, max_w_ratio = 0.15, 0.50
    min_h_ratio, max_h_ratio = 0.08, 0.35
    default_w_ratio = 0.35

    # 计算允许的比例范围（以保持在画面内）
    ratio_max = min((w_cup * max_w_ratio) / max(1, w_logo), (h_cup * max_h_ratio) / max(1, h_logo))
    ratio_min_target = max((w_cup * min_w_ratio) / max(1, w_logo), (h_cup * min_h_ratio) / max(1, h_logo))
    ratio_min = min(ratio_min_target, ratio_max)

    # 期望比例：使用传入 scale 或默认值，然后在范围内裁剪
    if scale is None:
        desired_ratio = (w_cup * default_w_ratio) / max(1, w_logo)
    else:
        desired_ratio = (w_cup * float(scale)) / max(1, w_logo)
    ratio = max(ratio_min, min(desired_ratio, ratio_max))

    target_w = max(1, int(w_logo * ratio))
    target_h = max(1, int(h_logo * ratio))
    logo_resized = cv2.resize(logo_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 圆柱弯曲
    if curve > 0:
        logo_resized = cylindrical_warp(logo_resized, curve=curve)

    # 提取 mask
    b,g,r,a = cv2.split(logo_resized)
    mask = a.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (7,7), 3)  # 柔化边缘

    # logo 颜色部分
    logo_rgb = cv2.merge([b,g,r])

    # 放置位置：默认将 logo 的中心放在杯子高度的下 2/3 处，并水平居中
    if position == "center":
        center_x = w_cup // 2
        center_y = int(h_cup * (2.0/3.0))
        x = center_x - logo_rgb.shape[1] // 2
        y = center_y - logo_rgb.shape[0] // 2
    else:
        x, y = position

    # ROI
    # 保证不越界（如靠近底部时）
    x = max(0, min(x, w_cup - logo_rgb.shape[1]))
    y = max(0, min(y, h_cup - logo_rgb.shape[0]))

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_cup, x + logo_rgb.shape[1])
    y2 = min(h_cup, y + logo_rgb.shape[0])

    roi = cup[y1:y2, x1:x2]
    logo_part = logo_rgb[0:y2-y1, 0:x2-x1]
    mask_part = mask[0:y2-y1, 0:x2-x1]

    # alpha 融合
    blended = (roi * (1 - alpha*mask_part[...,None]) + logo_part * (alpha*mask_part[...,None])).astype(np.uint8)
    cup[y1:y2, x1:x2] = blended

    return cup

if __name__ == "__main__":
    cup = cv2.imread("../images/cup.png", cv2.IMREAD_COLOR)
    logo = cv2.imread("../images/logo22.png", cv2.IMREAD_UNCHANGED)


    result = place_logo(cup, logo, scale=0.3, alpha=0.9, curve=0.25)

    cv2.imwrite("cup_logo_result.png", result)
    print("✅ 已保存 cup_logo_result.png")

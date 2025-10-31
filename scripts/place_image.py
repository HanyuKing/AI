import cv2
import numpy as np

def warp_logo_cylinder_final(logo, curve_strength=0.3):
    """
    最终圆柱弯曲函数，水平收缩模拟圆柱贴合，垂直保持不变。
    curve_strength: 弯曲强度，0~0.5，越大弯曲越明显
    """
    h, w = logo.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    cx = w / 2
    for y in range(h):
        for x in range(w):
            theta = (x - cx) / cx * (np.pi/2) * curve_strength  # 左右收缩弧度
            x_new = cx + np.sin(theta) * cx
            map_x[y, x] = np.clip(x_new, 0, w-1)
            map_y[y, x] = y  # 垂直不变
    warped = cv2.remap(logo, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

def logo_bottom_center_pro_final(cup_path, logo_path, output_path,
                                 scale=0.3, margin=20, curve_strength=0.3):
    """
    将logo贴在杯子下方中间（最终专业版本）
    功能：
    - 圆柱弯曲（可见）
    - 光照融合
    - alpha混合
    - 自动下方中间贴合
    """
    # 1️⃣ 读取图像
    cup = cv2.imread(cup_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    if cup is None or logo is None:
        raise ValueError("图片读取失败，请检查路径")

    # 2️⃣ 调整logo大小
    new_w = int(cup.shape[1] * scale)
    new_h = int(logo.shape[0] * (new_w / logo.shape[1]))
    logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3️⃣ 圆柱弯曲
    logo = warp_logo_cylinder_final(logo, curve_strength)

    # 4️⃣ alpha通道
    if logo.shape[2] == 4:
        alpha = logo[..., 3] / 255.0
        logo_rgb = logo[..., :3]
    else:
        alpha = np.ones((new_h, new_w))
        logo_rgb = logo

    # 5️⃣ 自动计算下方中间位置
    x = (cup.shape[1] - new_w) // 2
    y = cup.shape[0] - new_h - margin
    if y < 0:
        raise ValueError("logo太大或margin太大，超出杯子范围")

    # 6️⃣ 光照融合
    gray = cv2.cvtColor(cup, cv2.COLOR_BGR2GRAY)
    light_map = cv2.normalize(gray, None, 0.6, 1.2, cv2.NORM_MINMAX)
    light_map = cv2.cvtColor(light_map, cv2.COLOR_GRAY2BGR)
    light_roi = light_map[y:y+new_h, x:x+new_w]

    logo_rgb = logo_rgb.astype(np.float32) / 255.0
    logo_rgb = np.clip(logo_rgb * light_roi, 0, 1)

    # 7️⃣ alpha混合
    roi = cup[y:y+new_h, x:x+new_w].astype(np.float32) / 255.0
    blended = roi * (1 - alpha[..., None]) + logo_rgb * alpha[..., None]
    cup[y:y+new_h, x:x+new_w] = (blended * 255).astype(np.uint8)

    # 8️⃣ 保存结果
    cv2.imwrite(output_path, cup)
    print(f"✅ 已生成最终专业 Mockup效果: {output_path}")
# === 示例调用 ===
logo_bottom_center_pro_final(
    cup_path="../images/cup.png",
    logo_path="../images/logo1.png",
    output_path="cup_bottom_center.png",
    scale=0.4,           # logo宽度占杯子比例
    margin=20,           # 底部距离
    curve_strength=0.8   # 弧形强度，越小越平整
)
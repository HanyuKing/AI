from typing import Optional, Tuple

import cv2
import numpy as np

def create_rotation_matrix(angle: float) -> np.ndarray:
    """创建2D旋转矩阵

    Args:
        angle: 旋转角度（弧度），正值为逆时针

    Returns:
        2x2旋转矩阵
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    return np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])


def get_rotated_bounds(width: int, height: int, angle: float) -> Tuple[int, int]:
    """计算旋转后图像的边界尺寸

    Args:
        width: 原图宽度
        height: 原图高度
        angle: 旋转角度（弧度）

    Returns:
        (new_width, new_height) 新的图像尺寸
    """
    # 图像四个角的坐标
    corners = np.array([
        [-width/2, -height/2],
        [width/2, -height/2],
        [width/2, height/2],
        [-width/2, height/2]
    ])

    # 应用旋转矩阵
    rotation_matrix = create_rotation_matrix(angle)
    rotated_corners = corners @ rotation_matrix.T

    # 计算新的边界
    min_x, min_y = np.min(rotated_corners, axis=0)
    max_x, max_y = np.max(rotated_corners, axis=0)

    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    return new_width, new_height

def rotate_image_numpy(img: np.ndarray,
                       angle_degrees: float,
                       center: Optional[Tuple[float, float]] = None,
                       expand: bool = True,
                       fill_value: int = 0,
                       interpolation: str = 'bilinear') -> np.ndarray:
    """使用NumPy实现图像旋转（带内存统计）

    Args:
        img: 输入图像
        angle_degrees: 旋转角度（度），正值为逆时针
        center: 旋转中心，None表示图像中心
        expand: 是否扩展画布以容纳完整旋转图像
        fill_value: 填充值
        interpolation: 插值方法 ('nearest', 'bilinear')

    Returns:
        旋转后的图像
    """
    def get_memory_size(var, name):
        """获取变量的内存大小"""
        if isinstance(var, np.ndarray):
            size_mb = var.nbytes / (1024 * 1024)
            print(f"  {name}: {var.shape} {var.dtype} -> {size_mb:.2f} MB")
            return size_mb
        else:
            return 0

    print(f"\n=== 图像旋转内存统计 (角度: {angle_degrees}°) ===")

    height, width = img.shape[:2]
    original_memory = get_memory_size(img, "原始图像")
    total_memory = original_memory

    # 转换为弧度
    angle_rad = np.radians(angle_degrees)

    # 设置旋转中心
    if center is None:
        center = (width / 2.0, height / 2.0)

    cx, cy = center

    # 创建旋转矩阵
    rotation_matrix = create_rotation_matrix(-angle_rad)  # 负号实现顺时针旋转
    matrix_memory = get_memory_size(rotation_matrix, "旋转矩阵")
    total_memory += matrix_memory

    if expand:
        # 计算新的图像尺寸
        new_width, new_height = get_rotated_bounds(width, height, angle_rad)

        # 计算新的中心偏移
        offset_x = (new_width - width) / 2.0
        offset_y = (new_height - height) / 2.0
    else:
        new_width, new_height = width, height
        offset_x, offset_y = 0, 0

    # 创建输出图像
    if len(img.shape) == 3:
        rotated_img = np.full((new_height, new_width, img.shape[2]),
                              fill_value, dtype=img.dtype)
    else:
        rotated_img = np.full((new_height, new_width),
                              fill_value, dtype=img.dtype)

    output_memory = get_memory_size(rotated_img, "输出图像")
    total_memory += output_memory

    # 创建坐标网格 (使用float32减少内存)
    y_coords, x_coords = np.mgrid[0:new_height, 0:new_width]
    coords_memory = get_memory_size(y_coords, "Y坐标网格") + get_memory_size(x_coords, "X坐标网格")
    total_memory += coords_memory

    # 将坐标转换到原图像坐标系
    # 1. 直接转换为float32并中心化，避免创建额外的float64数组
    x_coords = x_coords.astype(np.float32) - new_width / 2.0
    y_coords = y_coords.astype(np.float32) - new_height / 2.0

    centered_memory = get_memory_size(x_coords, "X中心化坐标") + get_memory_size(y_coords, "Y中心化坐标")
    total_memory += centered_memory

    # 2. 应用逆旋转
    coords = np.stack([x_coords.ravel(), y_coords.ravel()])
    coords_stack_memory = get_memory_size(coords, "坐标堆栈")
    total_memory += coords_stack_memory

    # 释放不再需要的数组
    del x_coords, y_coords
    print(f'释放内存: {centered_memory:.2f} MB')
    total_memory -= centered_memory

    # 转换旋转矩阵为float32
    rotation_matrix_f32 = rotation_matrix.T.astype(np.float32)
    original_coords = rotation_matrix_f32 @ coords
    original_coords_memory = get_memory_size(original_coords, "变换后坐标")
    total_memory += original_coords_memory

    # 释放坐标堆栈
    del coords
    print(f'释放内存: {coords_stack_memory:.2f} MB')
    total_memory -= coords_stack_memory

    # 3. 移动到原图像坐标系
    original_x = original_coords[0] + cx
    original_y = original_coords[1] + cy

    # 重新整形
    original_x = original_x.reshape(new_height, new_width)
    original_y = original_y.reshape(new_height, new_width)

    # 释放变换后坐标
    del original_coords
    print(f'释放内存: {original_coords_memory:.2f} MB')
    total_memory -= original_coords_memory

    original_xy_memory = get_memory_size(original_x, "原始X坐标") + get_memory_size(original_y, "原始Y坐标")
    total_memory += original_xy_memory

    # 插值
    if interpolation == 'nearest':
        print("  使用最近邻插值...")
        # 最近邻插值 (使用int32减少内存)
        x_int = np.round(original_x).astype(np.int32)
        y_int = np.round(original_y).astype(np.int32)

        int_coords_memory = get_memory_size(x_int, "整数X坐标") + get_memory_size(y_int, "整数Y坐标")
        total_memory += int_coords_memory

        # 释放原始坐标
        del original_x, original_y
        print(f'释放内存: {original_xy_memory:.2f} MB')
        total_memory -= original_xy_memory

        # 边界检查
        valid_mask = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
        mask_memory = get_memory_size(valid_mask, "有效掩码")
        total_memory += mask_memory

        # 应用插值
        rotated_img[valid_mask] = img[y_int[valid_mask], x_int[valid_mask]]

        # 释放临时数组
        del x_int, y_int, valid_mask
        print(f'释放内存: {int_coords_memory:.2f} MB')
        total_memory -= int_coords_memory
        print(f'释放内存: {mask_memory:.2f} MB')
        total_memory -= mask_memory


    elif interpolation == 'bilinear':
        print("  使用双线性插值...")
        # 双线性插值 (使用float32减少内存)
        x0 = np.floor(original_x).astype(np.int32)
        y0 = np.floor(original_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # 计算权重 (使用float32)
        wx = (original_x - x0).astype(np.float32)
        wy = (original_y - y0).astype(np.float32)

        # 释放原始坐标
        del original_x, original_y

        # 边界检查
        valid_mask = (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)

        # 应用双线性插值
        if len(img.shape) == 3:
            for c in range(img.shape[2]):
                rotated_img[valid_mask, c] = (
                        img[y0[valid_mask], x0[valid_mask], c] * (1 - wx[valid_mask]) * (1 - wy[valid_mask]) +
                        img[y0[valid_mask], x1[valid_mask], c] * wx[valid_mask] * (1 - wy[valid_mask]) +
                        img[y1[valid_mask], x0[valid_mask], c] * (1 - wx[valid_mask]) * wy[valid_mask] +
                        img[y1[valid_mask], x1[valid_mask], c] * wx[valid_mask] * wy[valid_mask]
                )
        else:
            rotated_img[valid_mask] = (
                    img[y0[valid_mask], x0[valid_mask]] * (1 - wx[valid_mask]) * (1 - wy[valid_mask]) +
                    img[y0[valid_mask], x1[valid_mask]] * wx[valid_mask] * (1 - wy[valid_mask]) +
                    img[y1[valid_mask], x0[valid_mask]] * (1 - wx[valid_mask]) * wy[valid_mask] +
                    img[y1[valid_mask], x1[valid_mask]] * wx[valid_mask] * wy[valid_mask]
            )

        # 释放临时数组
        del x0, y0, x1, y1, wx, wy, valid_mask

    # 计算总内存使用和内存放大倍数
    total_memory = original_memory + output_memory
    memory_multiplier = total_memory / original_memory

    print(f"  总内存使用: {total_memory:.2f} MB")
    print(f"  内存放大倍数: {memory_multiplier:.2f}x")
    print("=" * 50)

    return rotated_img

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """以原图中心为基准旋转图片（顺时针为正）"""

    height, width = img.shape[:2]

    # 计算旋转中心（图像中心）
    center = (width / 2.0, height / 2.0)

    # 创建旋转变换矩阵（负角度实现顺时针旋转）
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # 计算旋转后的边界框
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])

    # 计算新的图像尺寸
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))

    # 调整旋转矩阵的平移部分，使旋转后的图像在新画布中居中
    rotation_matrix[0, 2] += (new_width - width) / 2.0
    rotation_matrix[1, 2] += (new_height - height) / 2.0

    # 执行旋转变换
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

    return rotated_img

if __name__ == '__main__':
    image = cv2.imread("../images/a.png")
    rotated_image = rotate_image(image, 45)
    cv2.imwrite('../output/rotated_image.png', rotated_image)

    rotate_image_numpy = rotate_image_numpy(image, -45, interpolation="nearest")
    cv2.imwrite('../output/rotate_image_numpy.png', rotate_image_numpy)
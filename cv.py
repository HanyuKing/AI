import cv2

# 读取图像
image = cv2.imread("./images/a.png")  # 确保example.jpg在当前目录下

# 检查图像是否成功读取
if image is None:
    print("无法读取图像。请检查文件路径。")
    exit()

# 调整图像大小
resized_image = cv2.resize(image, (800, 600))
cv2.imwrite('./output/resized_image.png', resized_image)

# 获取图像中心
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# 定义旋转矩阵，旋转45度
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated_image = cv2.warpAffine(image, M, (w, h))
cv2.imwrite('./output/rotated_image.png', rotated_image)

# 水平翻转
flipped_image = cv2.flip(image, 1)
cv2.imwrite('./output/flipped_image.png', flipped_image)
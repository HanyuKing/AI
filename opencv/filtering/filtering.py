import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 添加高斯噪声进行演示
def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声"""
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def test_image_filtering():

    img = cv.imread('../data/opencv-logo-white.png')
    assert img is not None, "file could not be read, check with os.path.exists()"

    blur = cv.blur(img,(1,1))

    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

def test_image_blurring():
    img = cv.imread('../data/opencv-logo-white.png')
    assert img is not None, "file could not be read, check with os.path.exists()"

    blur = cv.blur(img,(5,5))

    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()


def test_image_median_blurring():
    img = cv.imread('../data/opencv-logo-white.png')
    assert img is not None, "file could not be read, check with os.path.exists()"

    # 添加噪声
    noisy_img = add_gaussian_noise(img)
    blur = cv.medianBlur(noisy_img,5)

    plt.subplot(131),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(noisy_img),plt.title('Gaussian noise')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(blur),plt.title('Median Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

def test_image_bilateral_filtering():
    img = cv.imread('../data/opencv-logo-white.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 添加噪声
    noisy_img = add_gaussian_noise(img)

    # 应用双边滤波
    # 参数: 输入图像, 邻域直径, sigmaColor, sigmaSpace
    bilateral_filtered = cv2.bilateralFilter(noisy_img,
                                             d=9,
                                             sigmaColor=75,
                                             sigmaSpace=75)

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img)
    plt.title('Gaussian noise')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(bilateral_filtered)
    plt.title('Bilateral Filtering')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
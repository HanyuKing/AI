import cv2


def test_resize_image():
    img = cv2.imread("../images/baji_bg.png", cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (1500, 1500), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("../output/baji_bg_resize_1500.png", img)


if __name__ == '__main__':
    test_resize_image()
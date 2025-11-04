import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def test_erosion():
    img = cv.imread('../data/j.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(img,kernel,iterations = 1)

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(erosion),plt.title('erosion Output')
    plt.show()

def test_dilation():
    img = cv.imread('../data/j.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.dilate(img,kernel,iterations = 1)

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(erosion),plt.title('Dilation Output')
    plt.show()

def test_opening():
    img = cv.imread('../data/j.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(erosion),plt.title('morphologyEx OPEN Output')
    plt.show()

def test_closing():
    img = cv.imread('../data/j.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(erosion),plt.title('morphologyEx CLOSE Output')
    plt.show()

def test_gradient():
    img = cv.imread('../data/j.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(erosion),plt.title('morphologyEx GRADIENT Output')
    plt.show()

def test_opening_closing_diff():
    img = cv.imread('../data/j.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    kernel = np.ones((5,5),np.uint8)
    img_top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    img_black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    plt.subplot(131),plt.imshow(img),plt.title('Input')
    plt.subplot(132),plt.imshow(img_top_hat),plt.title('TOPHAT Output')
    plt.subplot(133),plt.imshow(img_black_hat),plt.title('BLACKHAT Output')
    plt.show()



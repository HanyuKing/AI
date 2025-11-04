import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def test_scaling():
    img = cv.imread('../data/messi5.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"

    res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    cv.imshow('res', res)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # OR
    #
    # height, width = img.shape[:2]
    # res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)


def test_translation():
    img = cv.imread('../data/messi5.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows, cols = img.shape

    M = np.float32([[1, 0, 300], [0, 1, 50]])
    dst = cv.warpAffine(img, M, (cols, rows))

    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def test_rotation():
    img = cv.imread('../data/messi5.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows, cols = img.shape
    M = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), 90, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

def test_affine_transformation():
    img = cv.imread('../data/messi5.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols,ch = img.shape

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])

    M = cv.getAffineTransform(pts1,pts2)

    dst = cv.warpAffine(img,M,(cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def test_perspective_transformation():
    img = cv.imread('../data/sudoku.png')
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows,cols,ch = img.shape

    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M = cv.getPerspectiveTransform(pts1,pts2)

    dst = cv.warpPerspective(img,M,(300,300))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()


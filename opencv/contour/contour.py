import numpy as np
import cv2 as cv

def test1():
    im = cv.imread('../data/girl.png')
    assert im is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 画出所有轮廓
    cnt = contours[4]
    cv.drawContours(im, [cnt], 0, (0, 255, 0), 3)
    cv.imshow('Contours', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

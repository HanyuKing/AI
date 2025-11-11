import cv2

img = cv2.imread('../data/messi5.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
lower_reso = cv2.pyrDown(img)
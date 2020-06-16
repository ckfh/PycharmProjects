import cv2 as cv
import numpy as np
import glob


left_path = r'C:\Users\hyh\Desktop\left\left01.jpg'
right_path = r'C:\Users\hyh\Desktop\right\right01.jpg'


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

img1 = cv.imread(left_path)
img2 = cv.imread(right_path)



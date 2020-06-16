import cv2 as cv
import numpy as np

img = cv.imread(r'C:\Users\hyh\Desktop\machine learning\1.jpg',1)
rows,cols,channel = img.shape
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow("1",dst)
cv.waitKey(0)
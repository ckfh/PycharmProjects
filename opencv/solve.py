import cv2 as cv
import numpy as np
import math
import c_config
import SGBM


path = r'C:\Users\Administrator\Desktop\PSMNet-master\\5.png'
left_path = r'C:\Users\Administrator\Desktop\PSMNet-master\\10_L.png'
right_path = r'C:\Users\Administrator\Desktop\PSMNet-master\\10_R.png'
img = cv.imread(path,-1)

# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# img = img.astype(np.int32)
# print(img.dtype)
print(img)
# cv.imshow('disp',img)
# a = cv.reprojectImageTo3D(img,c_config.Q)

cv.waitKey(0)

SGBM.SGBM(left_path,right_path)

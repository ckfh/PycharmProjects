import numpy as np
import cv2 as cv

import config

left_path = r'C:\Users\hyh\Desktop\left1'
right_path = r'C:\Users\hyh\Desktop\right2'
left_img = cv.imread(left_path + '\\1.jpg')
left_gray = cv.cvtColor(left_img,cv.COLOR_BGR2GRAY)

right_img = cv.imread(right_path + '\\1.jpg')

right_gray = cv.cvtColor(right_img,cv.COLOR_BGR2GRAY)

left_gray = cv.remap(left_gray, config.left_map1, config.left_map2, cv.INTER_LINEAR)
right_gray = cv.remap(right_gray, config.right_map1, config.right_map2, cv.INTER_LINEAR)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
left_ret,left_corners = cv.findChessboardCorners(left_gray, (11, 8), None)
right_ret,right_corners = cv.findChessboardCorners(right_gray, (11, 8), None)
if (left_ret and right_ret):
    corners2 = cv.cornerSubPix(left_gray, left_corners, (5, 5), (-1, -1), criteria)
    corners3 = cv.cornerSubPix(right_gray, right_corners, (5, 5), (-1, -1), criteria)
    if [corners2]:
        left_corners = corners2
    if [corners3]:
        right_corners = corners3

print(left_corners[0],right_corners[0])
Q = config.Q
cx = Q[0][3]
cy = Q[1][3]
print("cx = ",cx)
print("cy =",cy)
f = Q[2][3]
E = Q[3][3]
Tx = Q[3][2]
print(f)
print(1/Tx)

xr = 435
xl = 466

w = (xl-xr)*Tx+E
print(w)
print(f/w)

cv.circle(right_gray,(435,398),10,(0,0,255))
cv.circle(left_gray,(466,395),10,(0,0,255))

cv.imshow("left",left_gray)
cv.imshow("right",right_gray)
cv.waitKey(0)


#print(left_corners)#443.55692 362.68536 456.63354 349.74615
# dst = cv.drawChessboardCorners(gray, (11,8), left_corners,ret)
# cv.imshow("dst",dst)
# cv.waitKey(0)
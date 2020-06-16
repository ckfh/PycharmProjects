'''
@project : computeAngle
@author  : Hu
#@description: 简单计算二维角度
#@time   : 2019.11.7
'''

import cv2 as cv
import numpy as np

import _xihua


def Thin(image, array):
    h, w = image.shape
    iThin = image

    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                a = [1] * 9
                for k in range(3):
                    for l in range(3):
                        # 如果3*3矩阵的点不在边界且这些值为零，也就是黑色的点
                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and iThin[i - 1 + k, j - 1 + l] == 0:
                            a[k * 3 + l] = 0
                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                iThin[i, j] = array[sum] * 255
    return iThin


# 映射表
array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,\
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,\
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,\
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]


def onThresh(x):
    on_img = gussian.copy()
    blockSize = cv.getTrackbarPos("blockSize","thresh")
    print(blockSize)
    C = cv.getTrackbarPos("C","thresh")
    print(C)

    dst = cv.adaptiveThreshold(on_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

    cv.imshow("dst",dst)

def getCorners(x):
    ithin1 = ithin.copy()
    maxCorners = cv.getTrackbarPos("maxCorners","Corners")
    #qualityLevel = cv.getTrackbarPos("qualityLevel","Corners")
    minDistance = cv.getTrackbarPos("minDistance","Corners")
    print(minDistance)
    corners = cv.goodFeaturesToTrack(ithin1, maxCorners, 0.01, minDistance)

    corners = np.int32(corners)

    for i in corners:
        x, y = i.ravel()

        cv.circle(ithin1, (x, y), 3, 100, -1)

    cv.imshow("dst", ithin1)
    cv.waitKey(0)

img_path= r'C:\Users\hyh\Desktop\left'

img = cv.imread(img_path + '\\12.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# enhance = cv.equalizeHist(gray)
gussian = cv.GaussianBlur(gray,(5,5),0)
cv.imshow("gussian",gussian)

#自适应二值化,参数自调
# kernel = np.ones((5, 5), np.uint8)
# cv.namedWindow("thresh")
# blockSize = 20
# C = 2
# cv.createTrackbar("blockSize","thresh",blockSize,100,onThresh)
# cv.createTrackbar("C","thresh",C,20,onThresh)
# cv.imshow("thresh",gussian)
# cv.waitKey(0)
#
dst = cv.adaptiveThreshold(gussian, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 43, 7)
#
# cv.imshow("gussian",dst)
# cv.waitKey(0)

#细化
ithin = _xihua.Xihua(dst,array)

cv.imshow("ithin",ithin)
cv.waitKey(0)


#获取角点
corners = cv.goodFeaturesToTrack(ithin, 3, 0.01, 10)

#亚像素角点
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
sub_corners = cv.cornerSubPix(ithin,corners,(5,5),(-1,-1),criteria)
#显示角点并记住角点出现顺序
for i in sub_corners:
    x, y = i.ravel()
    print(x,y)
    cv.circle(ithin, (x, y), 10, 100, -1)
    cv.imshow("ithin",ithin)
    cv.waitKey(3000)

print(corners)
a = np.array([200-462,337-356])
b = np.array([200-362,337-152])
print(a.dot(b))
print(np.linalg.norm(a) , np.linalg.norm(b))
cosangle = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))
# print('A=({},{})'.format(str(57-227),str(469-318)))
# print('B=({},{})'.format(str(71-227),str(136-318)))
print("cos=",cosangle)
angle = np.arccos(cosangle)
print(np.degrees(angle))
cv.waitKey(0)
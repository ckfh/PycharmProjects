import numpy as np
import cv2 as cv

import _xihua
import c_config
import config


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

def callbackFunc(e, x, y, f, p):
    if e == cv.EVENT_LBUTTONDOWN:
        print(threeD[y][x])
def SGBM_update(val=0):
    global SGBM_num
    global SGBM_blockSize
    global threeD
    SGBM_blockSize=cv.getTrackbarPos('blockSize', 'SGNM_disparity')
    if SGBM_blockSize % 2 == 0:
        SGBM_blockSize += 1
    if SGBM_blockSize < 5:
        SGBM_blockSize = 5
    SGBM_stereo.setBlockSize(SGBM_blockSize)
    SGBM_num=cv.getTrackbarPos('num_disp', 'SGNM_disparity')
    num_disp = SGBM_num * 16
    SGBM_stereo.setNumDisparities(num_disp)

    SGBM_stereo.setUniquenessRatio(cv.getTrackbarPos('unique_Ratio', 'SGNM_disparity'))
    SGBM_stereo.setSpeckleWindowSize(cv.getTrackbarPos('spec_WinSize', 'SGNM_disparity'))
    SGBM_stereo.setSpeckleRange(cv.getTrackbarPos('spec_Range', 'SGNM_disparity'))
    SGBM_stereo.setDisp12MaxDiff(cv.getTrackbarPos('disp12MaxDiff', 'SGNM_disparity'))

    print('computing SGNM_disparity...')

    disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    threeD = cv.reprojectImageTo3D(disp.astype(np.float32) / 16., Q)
    cv.imshow('left', imgL)
    cv.imshow('right', imgR)
    cv.imshow('SGNM_disparity', (disp - min_disp) / num_disp)

def onThreshL(x):
    blockSize = cv.getTrackbarPos("blockSize","thresh")
    print(blockSize)
    C = cv.getTrackbarPos("C","thresh")
    print(C)

    dst = cv.adaptiveThreshold(imgL, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

    cv.imshow("dst",dst)

def onThreshR(x):
    blockSize = cv.getTrackbarPos("blockSize","thresh")
    print(blockSize)
    C = cv.getTrackbarPos("C","thresh")
    print(C)

    dst = cv.adaptiveThreshold(imgR, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

    cv.imshow("dst",dst)

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


def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv.LUT(img, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)
    return output_img



left_path = r'C:\Users\hyh\Desktop\left1'
right_path = r'C:\Users\hyh\Desktop\right2'

left_map1 = c_config.left_map1
left_map2 = c_config.left_map2
right_map2 = c_config.right_map2
right_map1 = c_config.right_map1

imgleft = cv.imread(left_path + '\\5.jpg')
imgleft = cv.cvtColor(imgleft,cv.COLOR_BGR2GRAY)
# imgleft = cv.equalizeHist(imgleft)
imgleft = cv.medianBlur(imgleft,5)
imgleft = gamma(imgleft,0.00000005, 4.0) #gamma变化，增强对比度
# imgleft = np.uint8(imgleft)
# cv.imshow("dst",imgleft)
# cv.waitKey(0)

imgright = cv.imread(right_path + '\\5.jpg')
imgright = cv.cvtColor(imgright,cv.COLOR_BGR2GRAY)
# imgright = cv.equalizeHist(imgright)
imgright = cv.medianBlur(imgright,5)
imgright = gamma(imgright,0.00000005, 4.0)


imgL = cv.remap(imgleft, left_map1, left_map2, cv.INTER_LINEAR)
imgR = cv.remap(imgright, right_map1, right_map2, cv.INTER_LINEAR)

imgL = cv.bitwise_not(imgL,None)
imgR = cv.bitwise_not(imgR,None)

# cv.imshow("dst",imgR)
# cv.waitKey(0)
# imgL = cv.resize(imgL,(640,480),cv.INTER_CUBIC)
# imgR = cv.resize(imgR,(640,480),cv.INTER_CUBIC)






# height,width=imgL.shape[:2]
# print(height,width)


# kernel = np.ones((5, 5), np.uint8)
# cv.namedWindow("thresh")
# blockSize = 20
# C = 2
# cv.createTrackbar("blockSize","thresh",blockSize,100,onThreshL)
# cv.createTrackbar("C","thresh",C,20,onThreshL)
# cv.imshow("thresh",imgL)
# cv.waitKey(0)

ret,dst = cv.threshold(imgR,248,255,cv.THRESH_BINARY)
# dst = cv.adaptiveThreshold(imgR, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 71,20)

#
# cv.imshow("dst",dst)
# cv.waitKey(0)
dst1 = cv.dilate(dst,None)
dst2 = cv.erode(dst,None)


ithin = Thin(dst1,array)
print(ithin.shape)

# cv.imshow("dst",ithin)
# cv.waitKey(0)

corners = cv.goodFeaturesToTrack(ithin, 3, 0.01, 100)

#亚像素角点
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
sub_corners = cv.cornerSubPix(ithin,corners,(5,5),(-1,-1),criteria)
#
# for cor in sub_corners:
#     print(cor[0])
#     cv.circle(ithin, tuple(cor[0]), 10, (0, 0, 255))
#     cv.imshow("1",ithin)
#     cv.waitKey(3000)
#
# cv.imshow("imgl",ithin)
# cv.waitKey(0)
Q = c_config.Q

cx = Q[0][3]
cy = Q[1][3]
print("cx = ",cx)
print("cy =",cy)
f = Q[2][3]
E = Q[3][3]
Tx = Q[3][2]
print(f)
print(Tx)
deep = np.array([560-452,479-130,259-197])
w = deep*Tx+E
Z = f/w
x = np.array([560,479,259])
y = np.array([273,116,333])

X = x - cx

Y = y - cy

T = np.vstack((X,Y,Z)).T
print("T= ",T)

BA= T[1] - T[0]
BC = T[1] - T[2]
print(BA,BC)
cosangle = BA.dot(BC)/(np.linalg.norm(BA) * np.linalg.norm(BC))
Lba = np.sum(np.linalg.norm(BA))
Lbc = np.sum(np.linalg.norm(BC))
print("Lba=",Lba)
print("Lbc=",Lbc)
print("cos=",cosangle)
angle = np.arccos(cosangle)
print(np.degrees(angle))


# t = np.sum(np.sqrt(t*t))






SGBM_blockSize = 5  # 一个匹配块的大小,大于1的奇数
SGBM_num = 2
min_disp = 0  # 最小的视差值，通常情况下为0
num_disp = SGBM_num * 16  # 192 - min_disp #视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
# blockSize = blockSize #匹配块大小（SADWindowSize），必须是大于等于1的奇数，一般为3~11
uniquenessRatio = 6  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
speckleRange = 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
speckleWindowSize = 60  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
disp12MaxDiff = 200  # 左右视差图的最大容许差异（超过将被清零），默认为 -1，即不执行左右视差检查。
P1 = 600  # 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
P2 = 2400  # p1控制视差平滑度，p2值越大，差异越平滑

cv.namedWindow('SGNM_disparity',2)
cv.setMouseCallback("SGNM_disparity", callbackFunc, None)
# cv.createTrackbar('blockSize', 'SGNM_disparity', SGBM_blockSize, 21, SGBM_update)
# cv.createTrackbar('num_disp', 'SGNM_disparity', SGBM_num, 20, SGBM_update)
# cv.createTrackbar('spec_Range', 'SGNM_disparity', speckleRange, 50, SGBM_update)  # 设置trackbar来调节参数
# cv.createTrackbar('spec_WinSize', 'SGNM_disparity', speckleWindowSize, 200, SGBM_update)
# cv.createTrackbar('unique_Ratio', 'SGNM_disparity', uniquenessRatio, 50, SGBM_update)
# cv.createTrackbar('disp12MaxDiff', 'SGNM_disparity', disp12MaxDiff, 250, SGBM_update)

# global SGBM_num
# global SGBM_blockSize
# global threeD
# SGBM_blockSize = cv.getTrackbarPos('blockSize', 'SGNM_disparity')
# if SGBM_blockSize % 2 == 0:
#     SGBM_blockSize += 1
# if SGBM_blockSize < 5:
#     SGBM_blockSize = 5
# SGBM_num = cv.getTrackbarPos('num_disp', 'SGNM_disparity')
# num_disp = SGBM_num * 16
#
# uniquenessRatio = cv.getTrackbarPos('unique_Ratio', 'SGNM_disparity')
# speckleWindowSize = cv.getTrackbarPos('spec_WinSize', 'SGNM_disparity')
# speckleRange = cv.getTrackbarPos('spec_Range', 'SGNM_disparity')
# disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'SGNM_disparity')
SGBM_stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,  # 最小的视差值
    numDisparities=num_disp,  # 视差范围
    blockSize=SGBM_blockSize,  # 匹配块大小（SADWindowSize）
    uniquenessRatio=uniquenessRatio,  # 视差唯一性百分比
    speckleRange=speckleRange,  # 视差变化阈值++++++
    speckleWindowSize=speckleWindowSize,
    disp12MaxDiff=disp12MaxDiff,  # 左右视差图的最大容许差异
    P1=P1,  # 惩罚系数
    P2=P2
)
# print('computing SGNM_disparity...')

disparity = SGBM_stereo.compute(imgL, imgR)
# disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U).astype(np.float32) / 16.0
disp = disparity.astype(np.float32) / 16.0
threeD = cv.reprojectImageTo3D(disp, c_config.Q)
cv.waitKey(0)
cv.imshow('left', imgL)
cv.imshow('right', imgR)
cv.imshow('SGNM_disparity', (disp - min_disp) / num_disp)
cv.waitKey(0)
import cv2 as cv
import numpy as np



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


def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2,P3,...,P8,P9,P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)


# Zhang-Suen 细化算法
def zhangSuen(image):
    Image_Thinned = image.copy()  # Making copy to protect original image
    changing1 = changing2 = 1
    while changing1 or changing2:  # Iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                                2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                            transitions(n) == 1 and  # Condition 2: S(P1)=1
                                    P2 * P4 * P6 == 0 and  # Condition 3
                                    P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                                2 <= sum(n) <= 6 and  # Condition 1
                            transitions(n) == 1 and  # Condition 2
                                    P2 * P4 * P8 == 0 and  # Condition 3
                                    P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
        return Image_Thinned

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

def onThresh(x):
    blockSize = cv.getTrackbarPos("blockSize","thresh")
    print(blockSize)
    C = cv.getTrackbarPos("C","thresh")
    print(C)

    dst = cv.adaptiveThreshold(imgleft, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

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
left_path = r'C:\Users\hyh\Desktop\left'
right_path = r'C:\Users\hyh\Desktop\right'
img = cv.imread(left_path + '\\12.jpg')
imgleft = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# imgleft = cv.equalizeHist(imgleft)
imgleft = cv.GaussianBlur(imgleft,(5,5),0)
kernel = np.ones((5, 5), np.uint8)
dst = cv.adaptiveThreshold(imgleft, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 43, 7)



ithin = Thin(dst,array)
# kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# cv.erode(ithin,kernel1,iterations=1)

corners = cv.goodFeaturesToTrack(ithin, 3, 0.01, 10)
# for i in corners:
#     x, y = i.ravel()
#     print(x,y)
#     cv.circle(ithin, (x, y), 3, 100, -1)
#     cv.imshow("ithin",ithin)
#     cv.waitKey(3000)

print(corners)
a = np.array([170,-151])
b = np.array([156,182])
print(a.dot(b))
print(np.linalg.norm(a) , np.linalg.norm(b))
cosangle = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))
print('A=({},{})'.format(str(57-227),str(469-318)))
print('B=({},{})'.format(str(71-227),str(136-318)))
print("cos=",cosangle)
angle = np.arccos(cosangle)
print(np.degrees(angle))
cv.waitKey(0)
#角点调参框
# maxCorners = 2
# #qualityLevel = 0.01
# minDistance = 10
# cv.namedWindow("Corners")
#
# cv.createTrackbar("maxCorners","Corners",maxCorners,10,getCorners)
# cv.createTrackbar("minDistance","Corners",minDistance,100,getCorners)
#
# cv.imshow("Corners",dst)
# cv.waitKey(0)
#---------------------------------

#创建调参框
# bolckSize = 0
# C = 0
# cv.namedWindow("thresh")
# cv.createTrackbar("blockSize","thresh",bolckSize,100,onThresh)
# cv.createTrackbar("C","thresh",C,10,onThresh)
# cv.imshow("thresh",imgleft)
# cv.waitKey(0)
#-------------


# ret,dst = cv.threshold(imgleft,100,255,cv.THRESH_BINARY)
#
# dst = cv.bitwise_not(dst) #取反
#
#
# ##zhang 细化
# BW_Skeleton = zhangSuen(dst)
# cv.imshow("1",BW_Skeleton)
# cv.waitKey(0)
#
# ##细化
#
# er = cv.erode(ithin,kernel,iterations=1)
# cv.imshow("1",er)
# cv.waitKey(0)
#
dst_block9_ksize19 = cv.cornerHarris(imgleft, 9, 19, 0.04)
dst = cv.dilate(dst_block9_ksize19,None)
ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
retval, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(imgleft,np.float32(centroids),(5,5),(-1,-1),criteria)
# print(centroids)
# print(corners)
#
# res = np.hstack((centroids,corners))
# res = np.int32(res)
for cor in corners:
    cv.circle(imgleft, tuple(cor), 10, (0, 0, 255))
# #----------------
# # cv.circle(imgleft,(1150,276),10,(0,0,255))
# # cv.circle(imgleft,(237,519),10,(0,0,255))
# # img[res[:,1],res[:,0]]=[0,0,255]
# # img[res[:,3],res[:,2]] = [0,255,0]
#
# # imgleft[dst_block9_ksize19 > 0.01 * dst_block9_ksize19.max()] = [100]
# #---------------------
cv.imshow("1",imgleft)
cv.waitKey(0)

imgright = cv.imread(right_path + '\\12.jpg')
imgright = cv.cvtColor(imgright,cv.COLOR_BGR2GRAY)
imgright = cv.equalizeHist(imgright)
imgright = cv.GaussianBlur(imgright,(5,5),0)



# left_camera_matrix = np.array([[824.93564, 0., 251.64723],
#                                [0., 825.93598, 286.58058],
#                                [0., 0., 1.]])
# left_distortion = np.array([[0.23233, -0.99375, 0.00160, 0.00145, 0.00000]])
#
#
#
# right_camera_matrix = np.array([[853.66485, 0., 217.00856],
#                                 [0., 852.95574, 269.37140],
#                                 [0., 0., 1.]])
# right_distortion = np.array([[0.30829, -1.61541, 0.01495, -0.00758, 0.00000]])
#
# om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
# R = cv.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# T = np.array([-70.59612, -2.60704, 18.87635]) # 平移关系向量
#
# size = (640, 480) # 图像尺寸
#
# # 进行立体更正
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_camera_matrix, left_distortion,
#                                                                   right_camera_matrix, right_distortion, size, R,
#                                                                   T)
# # 计算更正map
# left_map1, left_map2 = cv.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv.CV_16SC2)
# right_map1, right_map2 = cv.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv.CV_16SC2)


# print("Q = ",Q)
# cx = Q[0][3]
# cy = Q[1][3]
# print("cx = ",cx)
# print("cy =",cy)
# f = Q[2][3]
# E = Q[3][3]
# Tx = Q[3][2]
# print(f)
# print(1/Tx)
#
# xr = 1177
# xl = 1015
#
# w = (xl-xr)*Tx+E
# print(w)
# print(f/w)

# cv.waitKey(0)
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# ret,left_corners = cv.findChessboardCorners(imgright, (11, 8), None)
# print(ret)
# if (ret):
#     corners2 = cv.cornerSubPix(imgright, left_corners, (5, 5), (-1, -1), criteria)
#     if [corners2]:
#         left_corners = corners2
#
# img = cv.drawChessboardCorners(imgright, (11,8), left_corners,ret)
#
# # cv.circle(imgright,(983,716), 63, (0,0,255), -1)
# print(left_corners[0])
# cv.namedWindow('img',2)
# cv.imshow('img',img)
# cv.waitKey(0)






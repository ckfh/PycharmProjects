import numpy as np
import cv2 as cv


def callbackFunc(e, x, y, f, p):
    if e == cv.EVENT_LBUTTONDOWN:
        print(threeD[y][x])

def callbackFunc_left(e, x, y, f, p):
    if e == cv.EVENT_LBUTTONDOWN:
        print(x)

def callbackFunc_right(e, x, y, f, p):
    if e == cv.EVENT_LBUTTONDOWN:
        print(x)
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
img1 = cv.imread(r'C:\Users\hyh\Desktop\left1\left_19.jpg')
img2 = cv.imread(r'C:\Users\hyh\Desktop\right2\right_19.jpg')
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)


left_camera_matrix = np.array([[3304.78927,0.,674.21220],
                               [0.,3311.14718,541.99229],
                               [0.,0.,1.]])


left_distortion = np.array([[0.04677,-1.57066,-0.00004,0.00219,0.00000]])

right_camera_matrix = np.array([[3263.055910,621.91827],
                                [0,3269.28937,496.24898],
                                [0.,0.,1.]])


right_distortion = np.array([[-0.00937,0.49844,-0.00374,-0.00290,0.00000]])

om = np.array([-0.00905,0.05116,-0.00891])
R = cv.Rodrigues(om)[0]
T = np.array([ -212.74577,0.03935,-52.47263])

size = (640,480)#640,480



R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T,alpha=0)

f = Q[2][3]
Tx = 1/Q[3][2]
print(f,1/Tx)
# 计算更正map
left_map1, left_map2 = cv.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv.CV_16SC2)
#
#
left_recify = cv.remap(gray1,left_map1,left_map2,cv.INTER_LINEAR)
right_recify = cv.remap(gray2,right_map1,right_map2,cv.INTER_LINEAR)

ret1, corners1 = cv.findChessboardCorners(left_recify, (11, 8), None)
ret, corners2 = cv.findChessboardCorners(right_recify, (11, 8), None)
x1 = corners1[0][0][0]
x2 = corners2[0][0][0]
print(x1)
print(x2)
print(1/x1-x2)
cv.namedWindow("left")
cv.namedWindow("right")
cv.setMouseCallback("left", callbackFunc_left, None)
cv.setMouseCallback("right", callbackFunc_right, None)
cv.imshow("left",left_recify)
cv.imshow("right",right_recify)
cv.waitKey(0)

# for i in range(20,left_recify.shape[0],20):
#     cv.line(left_recify,(0,i),(left_recify.shape[1],i),(255,255,255))
#     cv.line(right_recify, (0, i), (right_recify.shape[1], i), (255, 255, 255))
#
#
#
# img_all = np.concatenate((left_recify,right_recify),axis=1)  #合并
#
# cv.imshow("1",img_all)
# cv.waitKey(0)

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
i = 0

# SGBM_blockSize = 5  # 一个匹配块的大小,大于1的奇数
# SGBM_num = 2
# min_disp = 0  # 最小的视差值，通常情况下为0
# num_disp = SGBM_num * 16  # 192 - min_disp #视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
# # blockSize = blockSize #匹配块大小（SADWindowSize），必须是大于等于1的奇数，一般为3~11
# uniquenessRatio = 6  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
# speckleRange = 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
# speckleWindowSize = 60  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
# disp12MaxDiff = 200  # 左右视差图的最大容许差异（超过将被清零），默认为 -1，即不执行左右视差检查。
# P1 = 600  # 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
# P2 = 2400  # p1控制视差平滑度，p2值越大，差异越平滑
#
# cv.namedWindow('SGNM_disparity')
# cv.setMouseCallback("SGNM_disparity", callbackFunc, None)
# cv.createTrackbar('blockSize', 'SGNM_disparity', SGBM_blockSize, 21, SGBM_update)
# cv.createTrackbar('num_disp', 'SGNM_disparity', SGBM_num, 20, SGBM_update)
# cv.createTrackbar('spec_Range', 'SGNM_disparity', speckleRange, 50, SGBM_update)  # 设置trackbar来调节参数
# cv.createTrackbar('spec_WinSize', 'SGNM_disparity', speckleWindowSize, 200, SGBM_update)
# cv.createTrackbar('unique_Ratio', 'SGNM_disparity', uniquenessRatio, 50, SGBM_update)
# cv.createTrackbar('disp12MaxDiff', 'SGNM_disparity', disp12MaxDiff, 250, SGBM_update)


cv.namedWindow("left")
cv.namedWindow("right")
cv.namedWindow("depth")
cv.namedWindow("thresh")
cv.moveWindow("left", 0, 0)
cv.moveWindow("right", 600, 0)
cv.createTrackbar("num", "depth", 0, 10, lambda x: None)
cv.createTrackbar("threshold","thresh",0,255,lambda x: None)

cv.setMouseCallback("depth", callbackFunc, None)

while cap.isOpened():
    ret, frame = cap.read()
    left_img = frame[:, 0:640, :]
    right_img = frame[:, 640:1280, :]
    if ret:
        img1_rectified = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
        img2_rectified = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

        imgL = cv.cvtColor(img1_rectified, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(img2_rectified, cv.COLOR_BGR2GRAY)




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
        # SGBM_stereo = cv.StereoSGBM_create(
        #     minDisparity=min_disp,  # 最小的视差值
        #     numDisparities=num_disp,  # 视差范围
        #     blockSize=SGBM_blockSize,  # 匹配块大小（SADWindowSize）
        #     uniquenessRatio=uniquenessRatio,  # 视差唯一性百分比
        #     speckleRange=speckleRange,  # 视差变化阈值++++++
        #     speckleWindowSize=speckleWindowSize,
        #     disp12MaxDiff=disp12MaxDiff,  # 左右视差图的最大容许差异
        #     P1=P1,  # 惩罚系数
        #     P2=P2
        # )
        # print('computing SGNM_disparity...')
        # disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        # threeD = cv.reprojectImageTo3D(disp, Q)
        # cv.imshow('left', imgL)
        # cv.imshow('right', imgR)
        # cv.imshow('SGNM_disparity', (disp - min_disp) / num_disp)
        # # num = cv.getTrackbarPos("num", "depth")
        # # blockSize = cv.getTrackbarPos("blockSize", "depth")
        # # if blockSize % 2 == 0:
        # #     blockSize += 1
        # # if blockSize < 5:
        # #     blockSize = 5
        # #
        # # stereo = cv.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
        # # disparity = stereo.compute(imgL, imgR)
        # #
        # #
        # # disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # #
        # # threeD = cv.reprojectImageTo3D(disparity.astype(np.float32) / 16.,Q)
        # #
        # # cv.imshow("left", img1_rectified)
        # # cv.imshow("right", img2_rectified)
        # # cv.imshow("depth", disp)
        # #
        # key = cv.waitKey(1)
        # if key == ord("q"):
        #     break


        num = cv.getTrackbarPos("num", "depth")
        thresh = cv.getTrackbarPos("threshold","thresh")
        blockSize = cv.getTrackbarPos("blockSize", "depth")
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5

        # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
        stereo = cv.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)


        disparity = stereo.compute(imgL, imgR)

        disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        thresh1,disp1 = cv.threshold(disp,thresh,255,cv.THRESH_BINARY)

        threeD = cv.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)

        cv.imshow("left", img1_rectified)
        cv.imshow("right", img2_rectified)
        cv.imshow("depth", disp)
        cv.imshow("thresh", disp1)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

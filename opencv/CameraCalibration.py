import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

right_path = r'C:\Users\hyh\Desktop\right2\*.jpg'
left_path = r'C:\Users\hyh\Desktop\left1\*.jpg'

left_images = glob.glob(left_path)
right_images = glob.glob(right_path)
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
def get_mtx_dist(path):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    images = glob.glob(path)
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
    global size
    i = 0
    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
        if (ret):
            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
        else:
            print(image)
    print(len(obj_points),len(img_points))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, size, None, None)
    # print("ret:", ret)
    # print("mtx:\n", mtx)  # 内参数矩阵
    # print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecs:\n", len(rvecs))  # 旋转向量  # 外参数
    # print("tvecs:\n", len(tvecs))  # 平移向量  # 外参数
    #
    # print("-----------------------------------------------------")

    return mtx, dist,rvecs,tvecs,img_points,size,obj_points


left_mtx,left_dist,left_r,left_t,left_imagesPoints,left_size,left_obj= get_mtx_dist(left_path)
right_mtx,right_dist,right_r,right_t,right_imagesPoints,right_size,right_obj= get_mtx_dist(right_path)

# print(left_size)
# cv.waitKey(0)

objp = np.zeros((11 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
rms = cv.stereoCalibrate(left_obj,left_imagesPoints,right_imagesPoints,left_mtx,left_dist,right_mtx,right_dist,left_size,flags=cv.CALIB_USE_INTRINSIC_GUESS,criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,30,1e-6))

R = rms[5]
T = rms[6]

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_mtx,left_dist,right_mtx,right_dist,left_size,R,T)

left_map1, left_map2 = cv.initUndistortRectifyMap(left_mtx,left_dist, R1, P1, left_size, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(left_mtx,left_dist, R2, P2, left_size, cv.CV_16SC2)

# img = cv.imread(images[2])
# h,w = img.shape[:2]
#
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
#
# dst = cv.undistort(img,mtx,dist,None,newcameramtx)
# x,y,w,h = roi
#
# dst1 = dst[y:y+h,x:x+w]
#
#
# cv.imshow("1",dst1)
#
#
# if cv.waitKey(0) & 0xff == ord(' '):
#     cv.destroyWindow('1')

imgleft = cv.imread(left_images[1])
imgleft = cv.cvtColor(imgleft,cv.COLOR_BGR2GRAY)

imgright = cv.imread(right_images[1])
imgright = cv.cvtColor(imgright,cv.COLOR_BGR2GRAY)

imgL = cv.remap(imgleft, left_map1, left_map2, cv.INTER_LINEAR)
imgR = cv.remap(imgright, right_map1, right_map2, cv.INTER_LINEAR)

# cv.resize(imgL,())

# for i in range(20,imgL.shape[0],20):
#     cv.line(imgL,(0,i),(imgL.shape[1],i),(255,255,255))
#     cv.line(imgR, (0, i), (imgR.shape[1], i), (255, 255, 255))
#
#
#
# img_all = np.concatenate((imgL,imgR),axis=1)  #合并
#
# cv.imshow("all",img_all)
# cv.waitKey(0)




# cv.namedWindow("depth")
# cv.createTrackbar("num", "depth", 0, 10, lambda x: None)
# cv.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)







print("1111111111111")

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
i = 0

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

cv.namedWindow('SGNM_disparity')
cv.setMouseCallback("SGNM_disparity", callbackFunc, None)
cv.createTrackbar('blockSize', 'SGNM_disparity', SGBM_blockSize, 21, SGBM_update)
cv.createTrackbar('num_disp', 'SGNM_disparity', SGBM_num, 20, SGBM_update)
cv.createTrackbar('spec_Range', 'SGNM_disparity', speckleRange, 50, SGBM_update)  # 设置trackbar来调节参数
cv.createTrackbar('spec_WinSize', 'SGNM_disparity', speckleWindowSize, 200, SGBM_update)
cv.createTrackbar('unique_Ratio', 'SGNM_disparity', uniquenessRatio, 50, SGBM_update)
cv.createTrackbar('disp12MaxDiff', 'SGNM_disparity', disp12MaxDiff, 250, SGBM_update)

while cap.isOpened():
    ret, frame = cap.read()
    left_img = frame[:, 0:640, :]
    right_img = frame[:, 640:1280, :]
    if ret:
        img1_rectified = cv.remap(left_img, left_map1, left_map2, cv.INTER_LINEAR)
        img2_rectified = cv.remap(right_img, right_map1, right_map2, cv.INTER_LINEAR)

        imgL = cv.cvtColor(img1_rectified, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(img2_rectified, cv.COLOR_BGR2GRAY)




        global SGBM_num
        global SGBM_blockSize
        global threeD
        SGBM_blockSize = cv.getTrackbarPos('blockSize', 'SGNM_disparity')
        if SGBM_blockSize % 2 == 0:
            SGBM_blockSize += 1
        if SGBM_blockSize < 5:
            SGBM_blockSize = 5
        SGBM_num = cv.getTrackbarPos('num_disp', 'SGNM_disparity')
        num_disp = SGBM_num * 16

        uniquenessRatio = cv.getTrackbarPos('unique_Ratio', 'SGNM_disparity')
        speckleWindowSize = cv.getTrackbarPos('spec_WinSize', 'SGNM_disparity')
        speckleRange = cv.getTrackbarPos('spec_Range', 'SGNM_disparity')
        disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff', 'SGNM_disparity')
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
        disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        threeD = cv.reprojectImageTo3D(disp, Q)
        cv.imshow('left', imgL)
        cv.imshow('right', imgR)
        cv.imshow('SGNM_disparity', (disp - min_disp) / num_disp)
        # num = cv.getTrackbarPos("num", "depth")
        # blockSize = cv.getTrackbarPos("blockSize", "depth")
        # if blockSize % 2 == 0:
        #     blockSize += 1
        # if blockSize < 5:
        #     blockSize = 5
        #
        # stereo = cv.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
        # disparity = stereo.compute(imgL, imgR)
        #
        #
        # disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        #
        # threeD = cv.reprojectImageTo3D(disparity.astype(np.float32) / 16.,Q)
        #
        # cv.imshow("left", img1_rectified)
        # cv.imshow("right", img2_rectified)
        # cv.imshow("depth", disp)
        #
        key = cv.waitKey(1)
        if key == ord("q"):
            break


# num = cv.getTrackbarPos("num", "depth")
# blockSize = cv.getTrackbarPos("blockSize", "depth")
# if blockSize % 2 == 0:
#     blockSize += 1
# if blockSize < 5:
#     blockSize = 5
#
# stereo = cv.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
# disparity = stereo.compute(imgL, imgR)
#
# disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
#
# threeD = cv.reprojectImageTo3D(disparity.astype(np.float32) / 16.,Q)
#
# cv.imshow("depth", disp)
#
# cv.waitKey(0)

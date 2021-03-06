'''
@project : binocular_vision
@author  : Hoodie_Willi
#@description: 储存左右摄像机参数
#@time   : 2019-04-09 19:34:22
'''
import cv2
import numpy as np

# #第一组参数
# #左摄像头参数
# # left_camera_matrix = np.array([[1394.26663, 0, 343.36581],
# #                                [0, 1393.25572, 206.56092],
# #                                [0, 0, 1]])
# left_camera_matrix = np.array([[414.669900747434, 0, 333.577366996918],
#                               [0, 398.202874116662,247.566754366858],
#                               [0, 0, 1]])
#
# # left_distortion = np.array([[-0.66451, 0.84137, -0.01001, -0.00274, 0]])
# left_distortion = np.array([[-0.0142459238653320, -0.0249570178218176, -0.000649836306243846, -0.00313533505626265, 0]])
#
# #右摄像头参数
# # right_camera_matrix = np.array([[1385.46346, 0, 344.38903],
# #                                 [0, 1385.09596, 197.18927],
# #                                 [0, 0, 1]])
# right_camera_matrix = np.array([[415.434729887572, 0, 362.193328109812],
#                                 [0, 399.397315970090, 252.992310082312],
#                                 [0, 0, 1]])
#
# # right_distortion = np.array([[-0.63339, -0.68796, -0.00491, -0.00675, 0]])
# right_distortion = np.array([[-0.00498976782370186,	-0.0324531590890297, -0.00287605698805147, -0.00124834031769090, 0]])
#
# # om = np.array([0.00456, 0.01463, 0.00042])        # 旋转关系向量
# # R = cv2.Rodrigues(om)[0]                           # 使用Rodrigues变换将om变换为R
# # R = np.array([[0.999782590858406, -0.00515226326090622, -0.0202045836320361],
# #              [0.00514705653510220, 0.999986705860809, -0.000309694481018785],
# #              [0.0202059106569857, 0.000205633016385222, 0.999795818599770]])
# R = np.array([[0.999901221631019, -0.00516587865637030, 0.0130713686545343],
#              [0.00522142544928531, 0.999977468343127, -0.00421894801441166],
#              [-0.0130492795614409, 0.00428678245055778, 0.999905665450071]])
# # print(R)
# # T = np.array([-59.63351, -0.15514, -0.35781])      # 平移关系向量
# T = np.array([-57.6537975538066, 0.216669778808589, 0.330456732035273])

#左摄像头参数
# left_camera_matrix = np.array([[1394.26663, 0, 343.36581],
#                                [0, 1393.25572, 206.56092],
#                                [0, 0, 1]])
left_camera_matrix = np.array([[3.302910585696820e+03, 0, 6.686643643304093e+02],
                              [0, 3.302898688532874e+03, 5.242387052950863e+02],
                              [0, 0, 1]])

# k1, k2, k3, p1, p2
# left_distortion = np.array([[-0.66451, 0.84137, -0.01001, -0.00274, 0]])
left_distortion = np.array([[0.022651008177756, -0.049812690435085, 9.590146229512929e-04, 0.001857191892015, 0]])

#右摄像头参数
# right_camera_matrix = np.array([[1385.46346, 0, 344.38903],
#                                 [0, 1385.09596, 197.18927],
#                                 [0, 0, 1]])
right_camera_matrix = np.array([[3.288106652576944e+03, 0, 6.389133453354036e+02],
                                [0, 3.287192925043060e+03, 4.964044181877480e+02],
                                [0, 0, 1]])

# right_distortion = np.array([[-0.63339, -0.68796, -0.00491, -0.00675, 0]])
right_distortion = np.array([[-0.001284174892842, 0.147288347511619, -0.003052813275215, -0.001395077272298, 0]])

# om = np.array([0.00456, 0.01463, 0.00042])        # 旋转关系向量
# R = cv2.Rodrigues(om)[0]                           # 使用Rodrigues变换将om变换为R
# R = np.array([[0.999782590858406, -0.00515226326090622, -0.0202045836320361],
#              [0.00514705653510220, 0.999986705860809, -0.000309694481018785],
#              [0.0202059106569857, 0.000205633016385222, 0.999795818599770]])
R = np.array([[0.993424224709156, -0.002586433218163, -0.114462308749302],
              [0.002659317086254, 0.999996346854210, 4.840566796816766e-04],
              [0.114460638621525, -7.852652051171433e-04, 0.993427473731681]])
# print(R)
# T = np.array([-59.63351, -0.15514, -0.35781])      # 平移关系向量
T = np.array([-62.594870214896230, 1.486353135504069, -3.445086542130464])
size = (1280, 1024)# 图像尺寸

# 进行立体更正, bouguet标定方法
#
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, size, R, T,alpha=1)

# h,w = img.shape[:2]
#
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print(Q)
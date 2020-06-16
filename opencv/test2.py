import numpy as np
import cv2 as cv
import direction
import gaosi_steger
import gray
import c_config
import Thin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import jixian
import miniConfig
import steger


def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv.LUT(img, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)
    return output_img
def Median(img):
    new_img = np.zeros(img.shape,dtype=np.uint8)
    for i in range(img.shape[1]):
        first = 0
        last = 0
        flag = False
        for j in range(1,img.shape[0]):
            current = img[j,i]
            if(current>200):
                if flag == False:
                    first = j
                    last = j
                    flag = True
                else:
                    last = j
        if flag:
            new_img[int((first+last)/2),i] = 255

    return new_img

def getPoint(img):
    flag = False
    top = [img.shape[0], img.shape[1]]
    low = [0, 0]
    left = []
    last = []
    leftcount = 0
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if (img[j, i] > 200):
                leftcount += 1
                if (top[0] > j):
                    top = [j, i]
                if (low[0] < j):
                    low = [j, i]
                if flag == False:
                    flag = True
                    left = [j, i]
                else:
                    last = [j, i]
    # for i in range(img.shape[1]):
    #     if(img[left[0],i]>200):
    #         last = [left[0],i]
    if top == left or top == last:
        mid = low
    else:
        mid = top

    return left,mid,last
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = np.math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down
def getCorners(gray):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_points = []
    objp = np.zeros((11 * 8, 3), np.float32)
    img_points = []
    ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
    if (ret):
        obj_points.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
    return img_points

def getPointsDist(p1,p2):
    Q = c_config.Q

    cx = Q[0][3]
    cy = Q[1][3]
    print("cx = ", cx)
    print("cy =", cy)
    f = Q[2][3]
    E = Q[3][3]
    Tx = 0.01667
    print("f=",f)
    print("-1/TX=",Tx)
    print("E=",E)
    deep = np.array([p1[0]-p2[0]])
    print("deep=", deep)
    w = deep * Tx + E
    Z = f / w

    print("Z=",Z)
def drawCircle(img):
    corners = cv.goodFeaturesToTrack(img, 3, 0.01, 20)

    # 亚像素角点
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    sub_corners = cv.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)

    for cor in sub_corners:
        print(cor[0])
        cv.circle(img, tuple(cor[0]), 10, 100)
        cv.imshow("222", img)
        cv.waitKey(3000)

    cv.imshow("imgl", img)
    cv.waitKey(0)

def getDecPoint(A1,B1,C1,img):
    flag1 = False
    flag2 = False
    mid = []
    left = []
    last = []
    for i in range(img.shape[1]):
        if(img[A1[0],i]>195):
            if flag1 == False:
                left = [A1[0],i]
                flag1 = True
        if (img[B1[0], i] > 200):
            if flag2 == False:
                mid = [B1[0], i]
                flag2 = True
        if (img[C1[0], i] > 195):

            last = [C1[0],i]
    flag = False
    top = [img.shape[0], img.shape[1]]
    low = [0, 0]
    leftcount = 0
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if (img[j, i] > 200):
                leftcount += 1
                if (top[0] > j):
                    top = [j, i]
                if (low[0] < j):
                    low = [j, i]
                if flag == False and left==[]:
                    flag = True
                    left = [j, i]
                elif last == []:
                    last = [j, i]
    if mid == []:
        if top == left or top == last:
            mid = low
        else:
            mid = top
    return left, mid, last

def getAngle(A1,B1,C1,A2,B2,C2):
    Q = miniConfig.Q
    print(Q)


    cx = Q[0][3]
    cy = Q[1][3]
    print("cx = ", cx)
    print("cy =", cy)
    f = Q[2][3]
    E = Q[3][3]
    Tx = Q[3][2]
    # Tx = 0.008333333
    # Tx = 0.01667
    print(f)
    print(Tx)
    print(E)
    print("1=",A1,B1,C1)
    print("2=", A2, B2, C2)
    deep = np.array([A1[1]-A2[1],B1[1]-B2[1] , C1[1]-C2[1]])
    print("deep=",deep)
    w = deep * Tx + E
    Z = f / w

    y = np.array([A1[0], B1[0], C1[0]])
    x = np.array([A1[1], B1[1], C1[1]])

    X = (x + cx)/w

    Y = (y + cy)/w

    T = np.vstack((X, Y, Z)).T
    print("T= ", T)


    print(T[0])
    print(T[1])
    print(T[2])
    A = np.array([T[0][0], T[0][1], T[0][2]])
    B = np.array([T[1][0], T[1][1], T[1][2]])
    # B = np.array([83.89963434,-33.14294243,238.204905])
    C = np.array([T[2][0], T[2][1], T[2][2]])
    BA = B-A
    BC = B-C
    print(BA, BC)
    print(A,B,C)

    cosangle = BA.dot(BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    # cosangle = cos_dist(BA,BC)
    Lba = np.linalg.norm(BA)
    Lbc = np.linalg.norm(BC)
    print("BA.BC=",BA.dot(BC))
    print("Lba=", Lba)
    print("Lbc=", Lbc)
    print("cos=", cosangle)
    angle = np.arccos(cosangle)
    print(np.degrees(angle))

    # ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    # #  将数据点分成三部分画，在颜色上有区分度
    # ax.scatter(T[0][0], T[0][1], T[0][2], c='y')  # 绘制数据点
    # ax.scatter(T[1][0], T[1][1], T[1][2], c='r')
    # ax.scatter(T[2][0], T[2][1], T[2][2], c='g')
    #
    # ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], [A[2], B[2], C[2]], label='parametric curve')
    #
    # ax.set_zlabel('Z')  # 坐标轴
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.show()
left_path = r'C:\Users\Administrator\Desktop\left'
right_path = r'C:\Users\Administrator\Desktop\right'

# left_map1 = c_config.left_map1
# left_map2 = c_config.left_map2
# right_map2 = c_config.right_map2
# right_map1 = c_config.right_map1

left_map1 = miniConfig.left_map1
left_map2 = miniConfig.left_map2
right_map2 = miniConfig.right_map2
right_map1 = miniConfig.right_map1

# left_back = cv.imread(left_path+'\\13.jpg')
# left_img = cv.imread(left_path+'\\14.jpg')
#
# right_back = cv.imread(left_path+'\\13.jpg')
# right_img = cv.imread(left_path+'\\14.jpg')

# new_left_img = cv.addWeighted(left_back,-1,left_img,1,0)
# new_right_img = cv.addWeighted(right_back,-1,right_img,1,0)
new_left_img = cv.imread(left_path+'\\9.jpg')
new_right_img = cv.imread(right_path + '\\9.jpg')


right_gray = cv.cvtColor(new_right_img,cv.COLOR_BGR2GRAY)
# right_gray = cv.resize(right_gray,(640,480))
right_median = cv.GaussianBlur(right_gray,(5,5),0)

left_gray = cv.cvtColor(new_left_img,cv.COLOR_BGR2GRAY)
# left_gray = cv.resize(right_gray,(640,480))
left_median = cv.GaussianBlur(left_gray,(5,5),0)

imgL = cv.remap(left_median, left_map1, left_map2, cv.INTER_LINEAR)
imgR = cv.remap(right_median, right_map1, right_map2, cv.INTER_LINEAR)

def Stereo1(lx,ly,rx,ry):
    m1 = miniConfig.left_camera_matrix.T.flatten()
    print(m1)
    m2 = miniConfig.right_camera_matrix.T.flatten()
    R = miniConfig.R.flatten()
    T = miniConfig.T.flatten()

    numerator = (m1[0] + m1[4]) / 2 * ((m2[0] + m2[4]) / 2 * T[0] - (rx - m2[6]) * T[2])
    denominator1 = (rx - m2[6]) * (R[6] * (lx - m1[6]) + R[7] * (ly - m1[7]) + R[8] * (m1[0] + m1[4]) / 2)
    denominator2 = (m2[0] + m2[4]) / 2 * (R[0] * (lx - m1[6]) + R[1] * (ly - m1[7]) + R[2] * (m1[0] + m1[4]) / 2)
    z = numerator / (denominator1 - denominator2)
    x = z * (lx - m1[6]) / ((m1[0] + m1[4]) / 2)
    y = z * (ly - m1[7]) / ((m1[0] + m1[4]) / 2)
    return np.array([x,y,z])

def getDepth(imgL,imgR):
    depth = np.zeros([imgR.shape[0],imgR.shape[1]])
    for i in range(imgL.shape[0]):
        for j in range(imgL.shape[1]):
            if(imgL[i,j]>200):
                for k in range(imgR.shape[1]):
                    if(imgR[i,k]>200 and depth[i,k] == 0):
                        depth[i,k] = abs(j-k)
                        break

    return depth
def getPt(left_points,right_points,index,flag):
    left_i = index
    right_i = index
    a = left_points[left_i]
    b = right_points[right_i]
    while(a[0] != b[0]):
        if(a[0] < b[0]):
            if(flag):
                left_i += 1
            else:
                left_i -=1
        else:
            if (flag):
                right_i += 1
            else:
                right_i -= 1
        a = left_points[left_i]
        b = right_points[right_i]
    return a,b
# new_left_img = cv.resize(new_left_img,(640,480))
# new_right_img = cv.resize(new_right_img,(640,480))

# imgL = cv.resize(imgL,(640,480))
# imgR = cv.resize(imgR,(640,480))


# imgL_points = getCorners(imgL)
# imgR_points = getCorners(imgR)
# print(imgL_points)
# print(imgR_points)


# p2 = [173, 196]
# p1 = [235, 196]
# getPointsDist(p1,p2)
#


#
# for i in range(20,imgL.shape[0],20):
#     cv.line(imgL,(0,i),(imgL.shape[1],i),(255,255,255))
#     cv.line(imgR, (0, i), (imgR.shape[1], i), (255, 255, 255))
#     cv.line(new_left_img, (0, i), (new_left_img.shape[1], i), (255, 255, 255))
#     cv.line(new_right_img, (0, i), (new_right_img.shape[1], i), (255, 255, 255))
# # # # # cv.circle(new_left_img,tuple([254, 71]),10,(255, 255, 255))
# # # # # cv.circle(new_right_img,tuple([233, 56]),10,(255, 255, 255))
# # # # # #
# # # # # #
# # # # # #
# cv.imshow('1',imgL)
# img_all = np.concatenate((imgL,imgR),axis=1)  #合并
# img_all1 = np.concatenate((new_left_img,new_right_img),axis=1)  #合并
# # # # #
# cv.imshow("all",img_all)
# cv.imshow("all1",img_all1)
# cv.waitKey(0)



# cv.imshow("th1",imgL)
# cv.imshow("th2",imgR)
# cv.waitKey(0)
# imgL = left_median
# imgR = cv.remap(right_median, right_map1, right_map2, cv.INTER_LINEAR)

# left_gamma = gamma(left_median,0.00000005, 4.1) #gamma变化，增强对比度
# cv.imshow("th1",left_gamma)
# cv.waitKey(0)
# imgL1 = imgL[200:420,350:930]
# imgR1 = imgR[200:420,350:930]
imgL1 = imgL[:420,200:900]
imgR1 = imgR[:420,200:900]
ret1,th1 = cv.threshold(imgL1,175,255,cv.THRESH_BINARY)
ret2,th2 = cv.threshold(imgR1,175,255,cv.THRESH_BINARY)

# th1 = th1[151:400,:]
# th2 = th2[151:400,:]
# cv.imshow("th1",th1)
# cv.imshow("th2",th2)
# cv.waitKey(0)

# #
# print(th1.shape)
# th1 =th1[200:,:]
# print(th1.shape)
# th2 = th2[200:,:]
# #
# cv.imshow("th1",th1)
# cv.imshow("th2",th2)
# cv.waitKey(0)
# #
# left_ = Median(th1)

#


# th1 = cv.bitwise_not(th1,None)

# xiahua = Thin.Thin(th1)
# xiahua = cv.bitwise_not(xiahua,None)
# #
# gray_method_left = gray.Cvpointgray(th1)
# gray_method_right = gray.Cvpointgray(th2)

# s1 = gaosi_steger.steger(th1)
# s2 = gaosi_steger.steger(th2)
dir_method_left,left_points = direction.direction(th1)
dir_method_right,right_points = direction.direction(th2)

left_img = dir_method_left.copy()
right_img = dir_method_right.copy()
# cv.imshow("th1",left_img)
# cv.imshow("th2",right_img)
# cv.waitKey(0)

# #获取角点
# left_corners = cv.goodFeaturesToTrack(left_img, 3, 0.01, 10)
# right_corners = cv.goodFeaturesToTrack(right_img, 3, 0.01, 10)
# #亚像素角点
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# left_sub_corners = cv.cornerSubPix(left_img,left_corners,(5,5),(-1,-1),criteria)
# right_sub_corners = cv.cornerSubPix(right_img,right_corners,(5,5),(-1,-1),criteria)
# #显示角点并记住角点出现顺序
# for i in left_sub_corners:
#     x, y = i.ravel()
#     print(x,y)
#     cv.circle(dir_method_left, (x, y), 5, 100, -1)
#     cv.imshow("left",dir_method_left)
# for i in right_sub_corners:
#     x, y = i.ravel()
#     print(x,y)
#     cv.circle(dir_method_right, (x, y), 5, 100, -1)
#     cv.imshow("right",dir_method_right)

# depth = getDepth(gray_method_left,gray_method_right)
# count = 0
# x=0
# y=0
# maxd = np.max(depth)
# for i in range(depth.shape[0]):
#     for j in range(depth.shape[1]):
#         if(depth[i,j]>=maxd):
#             x +=i
#             y +=j
#             count +=1
# x = x/count
# y = y/count
# print(x,'+',y)
# cv.circle(gray_method_left,(142,412),5,100)
# cv.circle(gray_method_right,(142,412),5,100)
# cv.circle(gray_method_left,(142,397),5,100)
# cv.circle(gray_method_right,(142,397),5,100)
# cv.imshow("oldth1",gray_method_left)
# cv.imshow("gray_method_left",gray_method_right)
#
# # cv.imshow("th2",th2)
# cv.waitKey(0)
# _steger_left = gaosi_steger.steger(gray_method_left)

# th1 = cv.bitwise_not(th1)
# zhuansuan_left = Thin.Thin(th1)
# zhuansuan_left = cv.bitwise_not(zhuansuan_left)
#
# cv.imshow("dstl",zhuansuan_left)
# cv.waitKey(0)
# gray_method_right= gray.Cvpointgray(th2)
# _steger_right = gaosi_steger.steger(gray_method_right)

# cv.circle(imgL, tuple([245,270]), 10, 100)
# cv.circle(imgR, tuple([189,270]), 10, 100)
# img_all1 = np.concatenate((imgL,imgR),axis=1)  #合并
# cv.imshow("dstl",img_all1)
# cv.waitKey(0)
# drawCircle(gray_method_left)


A1,B1,C1 = getPoint(left_img)
# A1 = np.array([270,245])
A2,B2,C2 = getDecPoint(A1,B1,C1,right_img)
# A3,B3,C3 = getPoint(gray_method_left)
# # A1 = np.array([270,245])
# A4,B4,C4 = getDecPoint(A1,B1,C1,gray_method_right)
# A1 = [57,494]
# B1 = [114,582]
# C1 = [77,665]
# A2 = [57,197]
# B2 = [114,256]
# C2 = [77,365]
# B1 = B3
# B2 = B4
A1,A2 = getPt(left_points,right_points,0,True)
C1,C2 = getPt(left_points,right_points,min(len(left_points),len(right_points))-1,False)
# B1 = [109, 502]
# B2 = [109, 173]
print(A1,B1,C1)
print(A2,B2,C2)
# cv.circle(gray_method_left,tuple(A1[::-1]),5,100)
# cv.circle(gray_method_left,tuple(B1[::-1]),5,100)
# cv.circle(gray_method_left,tuple(C1[::-1]),5,100)
# cv.circle(gray_method_right,tuple(A2[::-1]),5,100)
# cv.circle(gray_method_right,tuple(B2[::-1]),5,100)
# cv.circle(gray_method_right,tuple(C2[::-1]),5,100)
#
# cv.imshow("gray_method_left",gray_method_left)
# cv.imshow("gray_method_right",gray_method_right)
cv.circle(left_img,tuple(A1[::-1]),5,100)
cv.circle(left_img,tuple(B1[::-1]),5,100)
cv.circle(left_img,tuple(C1[::-1]),5,100)
cv.circle(right_img,tuple(A2[::-1]),5,100)
cv.circle(right_img,tuple(B2[::-1]),5,100)
cv.circle(right_img,tuple(C2[::-1]),5,100)
cv.imshow("dir_method_left",left_img)
cv.imshow("dir_method_right",right_img)
# A1 = [A1[0]+200,A1[1]+350]
# B1 = [B1[0]+200,B1[1]+350]
# C1 = [C1[0]+200,C1[1]+350]
# A2 = [A2[0]+200,A2[1]+350]
# B2 = [B2[0]+200,B2[1]+350]
# C2 = [C2[0]+200,C2[1]+350]
A1 = [A1[0],A1[1]+200]
B1 = [B1[0],B1[1]+200]
C1 = [C1[0],C1[1]+200]
A2 = [A2[0],A2[1]+200]
B2 = [B2[0],B2[1]+200]
C2 = [C2[0],C2[1]+200]



# B1 = [310, 860]
# B2 = [310, 528]
# A1,B1,C1 = [ 49,160],[115,161],[172,127]
# A2,B2,C2 = [0,168],[54,168],[102,135]


# new_img = np.zeros(gray_method_right.shape)
# cv.line(new_img,tuple(A2[::-1]),tuple(B2[::-1]),100)
# cv.line(new_img,tuple(B2[::-1]),tuple(C2[::-1]),100)
cv.circle(imgL,tuple(A1[::-1]),5,100)
cv.circle(imgL,tuple(B1[::-1]),5,100)
cv.circle(imgL,tuple(C1[::-1]),5,100)
cv.circle(imgR,tuple(A2[::-1]),5,100)
cv.circle(imgR,tuple(B2[::-1]),5,100)
cv.circle(imgR,tuple(C2[::-1]),5,100)

cv.imshow("1",imgL)
cv.imshow("2",imgR)

getAngle(A1,B1,C1,A2,B2,C2)


# cv.imshow("1",_steger)
# cv.imshow("2",th1)
cv.waitKey(0)





corners = cv.goodFeaturesToTrack(left_, 6, 0.01, 50)

#亚像素角点
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
sub_corners = cv.cornerSubPix(left_,corners,(5,5),(-1,-1),criteria)

for cor in sub_corners:
    print(cor)
    cv.circle(left_, tuple(cor[0]), 10, 100)
    cv.imshow("1",left_)
    cv.waitKey(3000)

cv.imshow("imgl",left_)
cv.waitKey(0)

th1 = cv.bitwise_not(th1,None)
xiahua = Thin.Thin(th1)
xiahua = cv.bitwise_not(xiahua,None)
# gray_left_img = gray.Cvpointgray(xiahua)



cv.imshow("gray_new_left",xiahua)
cv.waitKey(0)

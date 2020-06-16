import cv2 as cv
import numpy as np
# img = np.array([[1,2,3,4,5],
#                 [6,7,8,9,10],
#                 [1,2,3,4,5],
#                 [6,7,8,9,10],
#                 [1, 2, 3, 4, 5]])

# def gamma(img, c, v):
#     lut = np.zeros(256, dtype=np.float32)
#     for i in range(256):
#         lut[i] = c * i ** v
#     output_img = cv.LUT(img, lut) #像素灰度值的映射
#     output_img = np.uint8(output_img+0.5)
#     return output_img
# left_path = r'C:\Users\hyh\Desktop\left1'
# right_path = r'C:\Users\hyh\Desktop\right2'
# img1 = cv.imread(left_path + '\\5.jpg')
# gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
#
# gray = cv.medianBlur(gray,5)
# ret1,gray = cv.threshold(gray,120,255,cv.THRESH_BINARY)
# gray = gamma(gray,0.00000005, 4.0) #gamma变化，增强对比度


def steger(gray):
    m1 = np.array([[1, -1]])

    m2 = np.array([[1], [-1]])

    dx = cv.filter2D(gray, cv.CV_32FC1, m1)
    dy = cv.filter2D(gray, cv.CV_32FC1, m2)

    m3 = np.array([[1, -2, 1]])
    m4 = np.array([[1], [-2], [1]])
    m5 = np.array([[1, -1], [-1, 1]])

    dxx = cv.filter2D(gray, cv.CV_32FC1, m3)
    dyy = cv.filter2D(gray, cv.CV_32FC1, m4)
    dxy = cv.filter2D(gray, cv.CV_32FC1, m5)

    maxD = -1
    print(gray.shape)
    col = gray.shape[1]
    row = gray.shape[0]
    print(col,row)
    Pt = []
    pt = 0
    for i in range(col):
        for j in range(row):
            if gray[j, i] > 200:

                hessian = np.zeros((2, 2))
                hessian[0, 0] = dxx[j, i]
                hessian[0, 1] = dxy[j, i]
                hessian[1, 0] = dxy[j, i]
                hessian[1, 1] = dyy[j, i]

                eValue, eVectors = np.linalg.eig(hessian)

                fmaxD = 0

                if abs(eValue[0]) >= abs(eValue[1]):
                    nx = eVectors[0, 0]
                    ny = eVectors[0, 1]
                    fmaxD = eValue[0]

                else:
                    nx = eVectors[1, 0]
                    ny = eVectors[1, 1]
                    fmaxD = eValue[1]

                t = -(nx * dx[j, i] + ny * dy[j, i]) / (nx * nx * dxx[j, i] + 2 * nx * ny * dxy[j, i] + ny * ny * dyy[j, i])

                if abs(t * nx) <= 0.5 and abs(t * ny) <= 0.5:
                    Pt.append(i)
                    Pt.append(j)
    new_img = np.zeros(gray.shape,dtype=np.uint8)
    for k in range((int(len(Pt) / 2))):
        new_img[Pt[2 * k + 1], Pt[2 * k]] = 255
    return new_img









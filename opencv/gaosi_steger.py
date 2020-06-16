import numpy as np
import cv2 as cv
import math
def steger(gray):
    W = 3
    sigma = 1

    #
    # m2 = np.array([[1], [-1]])
    m1 = np.zeros((2 * W + 1, 2 * W + 1))
    m2 = np.zeros((2 * W + 1, 2 * W + 1))

    m3 = np.zeros((2*W+1,2*W+1))
    m4 = np.zeros((2*W+1,2*W+1))
    m5 = np.zeros((2*W+1,2*W+1))
    for i in range(-W,W):
        for j in range(-W,W):
            m1[i+W,j+W] = (-1/2*math.pi*pow(sigma,4))*i*math.exp(-1 * (i*i + j*j) / (2 * sigma*sigma))
            m2[i + W, j + W] = (-1 / 2 * math.pi * pow(sigma, 4)) * j * math.exp(-1 * (i * i + j * j) / (2 * sigma * sigma))
            m3[i+W,j+W] = (1 - (i*i) / (sigma*sigma))*math.exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(-1 / (2 * math.pi*pow(sigma, 4)))
            m4[i+W,j+W] = (1 - (j*j) / (sigma*sigma))*math.exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(-1 / (2 * math.pi*pow(sigma, 4)))
            m5[i+W,j+W] = ((i*j))*math.exp(-1 * (i*i + j*j) / (2 * sigma*sigma))*(1 / (2 * math.pi*pow(sigma, 6)))


    dx = cv.filter2D(gray, cv.CV_32FC1, m1)
    dy = cv.filter2D(gray, cv.CV_32FC1, m2)

    dxx = cv.filter2D(gray, cv.CV_32FC1, m3)
    dyy = cv.filter2D(gray, cv.CV_32FC1, m4)
    dxy = cv.filter2D(gray, cv.CV_32FC1, m5)


    col = gray.shape[1]
    row = gray.shape[0]
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
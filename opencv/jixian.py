import cv2 as cv
import numpy as np
import math

def jixian(gray):
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
    print(col, row)
    Pt = []

    flag = False
    for i in range(col):
        if flag == False:
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
                        flag = True
                        break

    x,y = int(Pt[0]),int(Pt[1])
    while(gray[y,x]>100):
        hessian = np.zeros((2, 2))
        hessian[0, 0] = dxx[y, x]
        hessian[0, 1] = dxy[y, x]
        hessian[1, 0] = dxy[y, x]
        hessian[1, 1] = dyy[y, x]

        eValue, eVectors = np.linalg.eig(hessian)

        fmaxD = 0
        if abs(eValue[0]) <= abs(eValue[1]):
            nx = eVectors[0, 0]
            ny = eVectors[0, 1]
            fmaxD = eValue[0]

        else:
            nx = eVectors[1, 0]
            ny = eVectors[1, 1]
            fmaxD = eValue[1]


        newx = x+nx
        newy = y+ny
        x = int(round(newx))
        y = int(round(newy))
        # print(nx,ny)
        # print(newx,newy)
        # print(x,y)
        # print(gray[y, x])



        print(x,y,00000)


        hessian = np.zeros((2, 2))
        hessian[0, 0] = dxx[y, x]
        hessian[0, 1] = dxy[y, x]
        hessian[1, 0] = dxy[y, x]
        hessian[1, 1] = dyy[y, x]

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


        t = -(nx * dx[y, x] + ny * dy[y, x]) / (nx * nx * dxx[y, x] + 2 * nx * ny * dxy[y, x] + ny * ny * dyy[y, x])

        x = x+t*nx
        y = y+t*ny

        x = int(round(x))
        y = int(round(y))

        print('a', x, y)

        Pt.append(x)
        Pt.append(y)


    new_img = np.zeros(gray.shape, dtype=np.uint8)
    for k in range((int(len(Pt) / 2))):
        new_img[Pt[2 * k + 1], Pt[2 * k]] = 255
    return new_img



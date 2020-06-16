import numpy as np
import cv2 as cv

def direction(img):
    K1 = np.array([[0,0,1,1,1,0,0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0]])
    K2 = np.array([[0,0,0,0,0,0,0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
    K3 = np.array([[1, 1, 1, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 1, 1, 1]])
    K4 = np.fliplr(K3)

    M1 = cv.filter2D(img, cv.CV_32FC1, K1)
    M2 = cv.filter2D(img, cv.CV_32FC1, K2)
    M3 = cv.filter2D(img, cv.CV_32FC1, K3)
    M4 = cv.filter2D(img, cv.CV_32FC1, K4)

    maxindex = 0
    maxvalue = 0
    x = 0
    y = 0
    Pt = []
    col = img.shape[1]
    row = img.shape[0]
    for i in range(col):
        flag = False
        for j in range(row):
            if img[j, i] >0:
                maxvalue = max(M1[j,i],M2[j,i],M3[j,i],M4[j,i])
                if(maxvalue>=maxindex):
                    x = i
                    y = j
                    flag = True
                    maxindex = maxvalue

        if(flag):
            Pt.append(x)
            Pt.append(y)

    new_img = np.zeros(img.shape, dtype=np.uint8)
    points = []
    for k in range((int(len(Pt) / 2))):
        new_img[Pt[2 * k + 1], Pt[2 * k]] = 255
        points.append([Pt[2 * k + 1],Pt[2 * k]])
    return new_img,points


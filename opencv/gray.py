import cv2
import numpy as np

import Thin
import _xihua
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
#图像预处理
# img1 = cv2.imread("C:\\Users\hyh\Desktop\\right2\\5.jpg")
# # img=img[100:400,450:1000]
# # cv2.imshow('img',img1)
# # cv2.waitKey(0)
# emptyImage3=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# ret1,th1 = cv2.threshold(emptyImage3,120,255,cv2.THRESH_BINARY)
# print(th1)
# cv2.imshow('img',th1)
# cv2.waitKey(0)
#灰度重心法提取光条纹公式函数
def Cvpointgray(img):
    new_img = np.zeros(img.shape,dtype=np.uint8)
    for i in range(img.shape[1]):
        sum_value = 0
        sum_valuecoor = 0
        current_value = []
        current_coordinat = []
        maxvalue = 0
        for l in range(img.shape[0]):
            maxvalue = max(img[l,i],maxvalue)

        for j in range(img.shape[0]):
            current = img[j,i]
            if(current>=maxvalue):
                current_value.append(current)
                current_coordinat.append(j)

        for k in range(len(current_value)):
            sum_valuecoor += current_value[k]*current_coordinat[k]
            sum_value += current_value[k]
        if sum_value !=0:
            x = sum_valuecoor / sum_value
            new_img[int(x),i] = 255
            # cv2.circle(img1,(i,int(x)),1,(0,255,0))

    # cv2.imshow("1",img1)
    # cv2.waitKey(0)
    return new_img

# th1 = cv2.bitwise_not(th1,None)
# xiahua = Thin.Thin(th1,array)
# xiahua = cv2.bitwise_not(xiahua,None)
# kernel = np.ones((1, 1), np.uint8)
# new_img = Cvpointgray(th1)
# new_img = cv2.dilate(new_img,kernel)
# cv2.imshow("new",new_img)
# cv2.waitKey(0)


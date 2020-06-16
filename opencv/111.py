import cv2
import numpy as np
import config

cap = cv2.VideoCapture(0)
floder_left = r"C:\Users\Administrator\Desktop\left"
floder_right = r"C:\Users\Administrator\Desktop\right   "
# ret = cap.set(3, 320)
# ret = cap.set(4, 240)
# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
i = 21
while cap.isOpened():
    ret, frame = cap.read()
    left_img = frame[:, 0:640, :]
    right_img = frame[:, 640:1280, :]
    if ret:
        # 显示两幅图片合成的图片
        # cv2.imshow('img', frame)
        # 显示左摄像头视图
        cv2.imshow('left', left_img)
        # 显示右摄像头视图
        cv2.imshow('right', right_img)
    key = cv2.waitKey(delay=2)
    if key == ord('t'):
        cv2.imwrite(floder_left + '\\'+str(i) + '.jpg', left_img)  #
        cv2.imwrite(floder_right + '\\' + str(i) + '.jpg', right_img)
        i += 1
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

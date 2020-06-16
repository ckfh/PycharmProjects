import numpy as np
import math
import cv2
pngfile = r'C:\Users\Administrator\Desktop\PSMNet-master\\000004_10L.png'
pngfile1 = r'C:\Users\Administrator\Desktop\PSMNet-master\\5.png'
img_depth = cv2.imread(pngfile1,cv2. IMREAD_ANYCOLOR)
img = cv2.imread(pngfile,cv2. IMREAD_ANYCOLOR)
rgb = np.array(img)
disp = np.array(img_depth)
rgbd = np.zeros((375, 1242, 4), dtype=np.uint8)
rgbd[:, :, 0] = rgb[:, :, 0]
rgbd[:, :, 1] = rgb[:, :, 1]
rgbd[:, :, 2] = rgb[:, :, 2]
rgbd[:, :, 3] = disp
print(rgbd.shape)
cv2.imwrite(r'C:\Users\Administrator\Desktop\PSMNet-master\\15.png',rgbd)
#apply colormap on deoth image(image must be converted to 8-bit per pixel first)
# im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
im_color=cv2.applyColorMap(img_depth,cv2.COLORMAP_JET)
cv2.imshow("1",im_color)

cv2.waitKey(0)

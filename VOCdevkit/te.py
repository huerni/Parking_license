import cv2
import numpy as np

path = "../VOCdevkit/VOC2007/test/"
image = cv2.imdecode(np.fromfile(path + "011875-87_92-265&465_452&540-458&531_278&548_278&480_458&463-0_0_31_29_31_26_26-94-20.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
# img = cv2.imread(path + "011875-87_92-265&465_452&540-458&531_278&548_278&480_458&463-0_0_31_29_31_26_26-94-20.jpg")
top = 466-10
bottom = 542+10
left = 265-10
right = 468+10
cropImg = image[top:bottom,left:right]
cv2.imshow("img", cropImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("../VOCdevkit",cropImg)
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:45:09 2019

@author: One
"""

from PIL import Image
from keras.models import load_model
import cv2
import numpy as np
characters="京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])



model = load_model("../cnn_plr/model_data/resnet34_model.h5")
print("导入模型完成")
print("读取图片")
pic = Image.open("../VOCdevkit/VOC2007/result/1.jpg")
pic.show()
#这里换两种方式是因为两种方式显示的通道顺序不同

image = cv2.imdecode(np.fromfile("../VOCdevkit/VOC2007/result/1.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
image = cv2.resize(image, ((224,224)))
cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # img = cv2.imencode("../License-Plate-Recognition-Items/冀DUQGGP.jpg")
img=image[np.newaxis,:,:,:]#图片是三维的但是训练时是转换成4维了所以需要增加一个维度
predict = model.predict(img)

print("车牌号为：", decode(predict))







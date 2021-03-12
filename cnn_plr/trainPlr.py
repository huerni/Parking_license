# 使用的是resnet34模型
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import add, Flatten
import numpy as np
import cv2
import os
import time
import Augmentor
from PIL import Image

seed = 7
np.random.seed(seed)

characters = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

width, height, n_len, n_class = 224, 224, 7, len(characters)

train_path = "../VOCdevkit/VOC2007/gen_res/"
val_path = "../VOCdevkit/VOC2007/gen_res_val/"
log_dir = "../cnn_plr/model_data/"

def get_data(path):
    n = np.random.randint(0, 31)
    t = np.random.randint(0, 2)
    # print(t)
    if t == 0:
        l = np.random.randint(0, len([lists for lists in os.listdir(path + str(n) + "/output") if os.path.isfile(os.path.join(path + str(n) + "/output", lists))])+1)
        dir_list = os.listdir(path + str(n) + "/output")
        if n < 10:
            label_name = dir_list[l][11:18]
        elif n >= 10:
            label_name = dir_list[l][12:19]
        img = cv2.imdecode(np.fromfile(path + str(n) + "/output/" + dir_list[l], dtype=np.uint8), cv2.IMREAD_COLOR)
    elif t == 1:
        l = np.random.randint(0, len([lists for lists in os.listdir(path + str(n)) if os.path.isfile(os.path.join(path + str(n), lists))])+1)
        dir_list = os.listdir(path + str(n))
        # print(l)
        # print(dir_list[l])
        img = cv2.imdecode(np.fromfile(path + str(n) + "/" + dir_list[l], dtype=np.uint8), cv2.IMREAD_COLOR)
        label_name = dir_list[l][:-4]
    # print(t, label_name)
    return img, label_name

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    # generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            b, a = get_data(train_path)
            # print(a)
            # a = G.genPlateString(-1, -1)
            # b = G.generate(a)
            # a = train_data[i][:-1]
            # b = cv2.imdecode(np.fromfile("../VOCdevkit/data_plr/train_images/"+train_data[i][:-1]+".jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(b, (224, 224))
            X[i] = img.astype('float32')
            for j, ch in enumerate(a):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def gen_val(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    # generator = ImageCaptcha(width=width, height=height)
    train_data, val_data = get_data()
    while True:
        for i in range(batch_size):
            b, a = get_data(val_path)
            # a = G.genPlateString(-1, -1)
            # b = G.generate(a)
            # a = val_data[i][:-1]
            # b = cv2.imdecode(np.fromfile("../VOCdevkit/data_plr/train_images/"+val_data[i][:-1]+".jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(b, (224, 224))
            X[i] = img.astype('float32')
            for j, ch in enumerate(a):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1

        yield X, y


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


"""
使用者三段代码测试生成结果，以及解码结果
X, y = next(gen(1))
plt.imshow(X[0])
print(decode(y))
plt.title(decode(y))
"""


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

# res_net网络
def train():
    inpt = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (56,56,64)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # (28,28,128)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # (14,14,256)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # (7,7,512)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    # x = Dense(1000,activation='softmax')(x)
    x = [Dense(n_class, activation='softmax', name='P%d' % (i + 1))(x) for i in range(7)]
    model = Model(inputs=inpt, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, save_freq=1)
    # 训练的时候每轮1000个样本共5轮，一个batch_size=32，所以一共有16W张图片
    model.fit(gen(), steps_per_epoch=1000, epochs=5,
              validation_data=gen_val(), validation_steps=1280)
    # model.fit_generator(gen(), samples_per_epoch=1000, nb_epoch=5,
    #                     nb_worker=1, pickle_safe=True,
    #                     validation_data=gen_val(), nb_val_samples=1280)

    model.save("../Parking_license/cnn_plr/model_data/resnet34_model.h5")

if __name__ == "__main__":
    # get_data(train_path)
    train()
    # gen()
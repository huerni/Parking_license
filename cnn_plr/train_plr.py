import tensorflow as tf
from keras import layers
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
from PIL import Image  # 注意Image,后面会用到

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H","J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"]

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64}

data_dir = "../VOCdevkit/data_plr/data/"
model_path = "../cnn_plr/model_data/"
# 读取数据
def get_data():
    x_data = []
    y_num = []
    for line2 in open("../VOCdevkit/data_plr/label.txt", encoding='utf8'):
        y, x = line2.split(',')
        x_data.append(x[1:-1])
        print("{}:{}".format(y[0], index[y[0]]))
        y_num.append(y)
    image_datas = []
    # for x in x_data:
    #     image = cv2.imread(data_dir + x)
    #     image = cv2.resize(image, (72,272))
    #     img = np.multiply(image, 1 / 255.0)
    #     image_datas.append(img)

    image_datas = np.array(image_datas)


def train_model(inputs, outputs):
    model = creat_model()
    # 配置优化器和损失函数
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    # 回调函数
    checkpoint = ModelCheckpoint(
        filepath=model_path+ 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        verbose=0,
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch')
    history = model.fit(
        inputs,
        outputs,
        epochs=30,
        callbacks=[checkpoint],
        verbose=1
    )
    model.save_weights(model_path + "trained_weights.h5")

def creat_model():
    # 输入层
    inputs = tf.keras.Input(shape=(72, 272, 3), name="inputs")
    # 卷积层-1
    layer1 = layers.Conv2D(32, (3, 3), activation=tf.nn.relu, name="conv-1")(inputs)
    # 卷积层-2
    layer2 = layers.Conv2D(32, (3, 3), activation=tf.nn.relu, name="conv-2")(layer1)
    # 最大池化层-1
    layer3 = layers.MaxPooling2D((2, 2), name="max-pooling-1")(layer2)
    # 卷积层-3
    layer4 = layers.Conv2D(64, (3, 3), activation=tf.nn.relu, name="conv-3")(layer3)
    # 卷积层-4
    layer5 = layers.Conv2D(64, (3, 3), activation=tf.nn.relu, name="conv-4")(layer4)
    # 最大池化层-2
    layer6 = layers.MaxPooling2D((2, 2), name="max-pooling-2")(layer5)
    # 卷积层-5
    layer7 = layers.Conv2D(128, (3, 3), activation=tf.nn.relu, name="conv-5")(layer6)
    # 卷积层-6
    layer8 = layers.Conv2D(128, (3, 3), activation=tf.nn.relu, name="conv-6")(layer7)
    # 全连接层-1
    layer9 = layers.MaxPooling2D((2, 2), name="max-pooling-3")(layer8)
    layer10 = layers.Flatten(name="fullc-1")(layer9)
    # 输出，全连接层-21~27
    outputs = [layers.Dense(65, activation=tf.nn.softmax, name="fullc-2{}".format(i + 1))(layer10) for i in range(7)]
    # 模型实例化
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PLR-CNN")

    return model

if __name__ == "__main__":
    get_data()
import Augmentor
import os

# 数据增强
train_path = "../VOCdevkit/VOC2007/gen_res/"
for i in range(30):
    p=Augmentor.Pipeline(train_path + str(i))
    # 随机区域擦除
    p.random_erasing(probability=1, rectangle_area=0.5)
    #弹性扭曲，类似区域扭曲的感觉
    p.random_distortion(probability=1, grid_height=5, grid_width=16, magnitude=8)
    # 错切变换
    p.shear(probability=1, max_shear_left=15, max_shear_right=15)
    p.sample(len([lists for lists in os.listdir(train_path + str(i)) if os.path.isfile(os.path.join(train_path + str(i), lists))]))
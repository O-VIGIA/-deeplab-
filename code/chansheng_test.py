import numpy as np
import os
from PIL import Image

with open("F:\code\HW\HWCV_Final\VOCdevkit\VOC2012\ImageSets\Segmentation//val.txt", 'r') as f:
    f_lines = f.read().splitlines()
max_num = len(f_lines)
test_id = np.random.randint(0, max_num, 100)
test_name = []
for i in test_id:
    test_name.append(f_lines[i])

input_dir_path = "F:\code\HW\HWCV_Final\VOCdevkit\VOC2012\JPEGImages/"
label_dir_path = "F:\code\HW\HWCV_Final\VOCdevkit\VOC2012\SegmentationClass/"

input_save_path = "F:\code\HW\HWCV_Final\img\input/"
label_save_path = "F:\code\HW\HWCV_Final\img\label/"

for name in test_name:
    image_path = input_dir_path + name + '.jpg'
    image = Image.open(image_path)
    image.save(input_save_path + name + '.jpg')
    label_path = label_dir_path + name + '.png'
    label = Image.open(label_path)
    label.save(label_save_path + name + '.png')




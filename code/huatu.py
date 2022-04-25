import matplotlib.pyplot as plt
import os
from PIL import Image

dir_origin_path = "img/input/"
label_origin_path = "img/label/"
dir_save_path = "img_out/"
img_names = os.listdir(dir_origin_path)
num = 0
for img_name in img_names[0:4]:
    num = num + 1
    image_path = os.path.join(dir_origin_path, img_name)
    label_path = os.path.join(label_origin_path, img_name[:-4] + '.png')
    predict_path = os.path.join(dir_save_path, img_name)
    image_ori = Image.open(image_path)
    label_ori = Image.open(label_path)
    r_image_ori = Image.open(predict_path)

    plt.subplot(3, 4, num)
    plt.title('Input')
    plt.imshow(image_ori)
    plt.axis('off')

    plt.subplot(3, 4, num+4)
    plt.title('Predict')
    plt.imshow(r_image_ori)
    plt.axis('off')

    plt.subplot(3, 4, num+8)
    plt.title('Label')
    plt.imshow(label_ori)
    plt.axis('off')

plt.show()
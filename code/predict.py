from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from deeplab import DeeplabV3
import torch
import torchvision

loader = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

Writer = SummaryWriter(log_dir='TB_log2')

if __name__ == "__main__":
    deeplab = DeeplabV3()
    mode = "dir_predict"
    test_interval = 100
    dir_origin_path = "img/input/"
    label_origin_path = "img/label/"
    dir_save_path = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image)
                r_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        num = 0
        for img_name in tqdm(img_names):
            num = num + 1
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                label_path = os.path.join(label_origin_path, img_name[:-4] + '.png')

                image_ori = Image.open(image_path)
                image = loader(image_ori.convert('RGB')).unsqueeze(0)

                label_ori = Image.open(label_path)
                label = loader(label_ori.convert('RGB')).unsqueeze(0)

                r_image_ori = deeplab.detect_image(image_ori)
                r_image = loader(r_image_ori.convert('RGB')).unsqueeze(0)

                # Writer.add_images('input', image, global_step=num)
                # Writer.add_images('predict', r_image, global_step=num)
                # Writer.add_images('label', label, global_step=num)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image_ori.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict' or 'dir_predict'.")

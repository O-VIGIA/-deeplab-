import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils_fit import fit_one_epoch

from torch.utils.tensorboard import SummaryWriter

Writer = SummaryWriter(log_dir="F:\code\HW\HWCV_Final\TB_log")

if __name__ == "__main__":
    Cuda = True
    num_classes = 21
    backbone = "mobilenet"
    pretrained = False
    model_path = "model_data/deeplab_mobilenetv2.pth"
    downsample_factor = 16
    input_shape = [512, 512]

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 4
    Freeze_lr = 5e-4

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 2
    Unfreeze_lr = 5e-5
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = False
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    Freeze_Train = True
    num_workers = 2

    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory("logs/")

    #   读取数据集对应的txt
    with open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        # Dataset
        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        # Training with freeze
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            mean_train_loss, mean_train_f_score, mean_val_loss, mean_val_f_score = fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          num_classes)

            # 可视化
            tags = ["train_loss", "train_f_score", "val_loss", "val_f_score", "learning_rate"]
            Writer.add_scalar(tags[0], mean_train_loss, epoch)
            Writer.add_scalar(tags[1], mean_train_f_score, epoch)
            Writer.add_scalar(tags[2], mean_val_loss, epoch)
            Writer.add_scalar(tags[3], mean_val_f_score, epoch)
            Writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            Writer.flush()

            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        # Dataset
        train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        # Training with NO freeze
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(start_epoch, end_epoch):
            mean_train_loss, mean_train_f_score, mean_val_loss, mean_val_f_score = fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          num_classes)
            # 可视化
            tags = ["train_loss", "train_f_score", "val_loss", "val_f_score", "learning_rate"]
            Writer.add_scalar(tags[0], mean_train_loss, epoch)
            Writer.add_scalar(tags[1], mean_train_f_score, epoch)
            Writer.add_scalar(tags[2], mean_val_loss, epoch)
            Writer.add_scalar(tags[3], mean_val_f_score, epoch)
            Writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            Writer.flush()

            lr_scheduler.step()

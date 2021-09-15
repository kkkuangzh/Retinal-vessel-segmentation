
# 准备DRIVE数据集
# cv2无法处理gif

import os
import h5py
import numpy as np
from PIL import Image


def write_hdf5(arr, outfile):
  with h5py.File(outfile, "w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


# DRIVE数据集路径
# 训练集
imgs_train = "./DRIVE/training/images/"
groundTruth_train = "./DRIVE/training/1st_manual/"
masks_train = "./DRIVE/training/mask/"
# 测试集
imgs_test = "./DRIVE/test/images/"
groundTruth_test = "./DRIVE/test/1st_manual/"
masks_test = "./DRIVE/test/mask/"

img_num = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_dataset/"


def get_datasets(img_path, gt_path, mask_path, train_test="null"):
    imgs = np.empty((img_num, height, width, channels))
    gts = np.empty((img_num, height, width))
    border_masks = np.empty((img_num, height, width))
    for path, subpath, files in os.walk(img_path):
        for i in range(len(files)):
            # 眼底图像
            img = Image.open(img_path+files[i])
            imgs[i] = np.asarray(img)

            # 对应专家标记
            gt_name = files[i][0:2] + "_manual1.gif"
            g_truth = Image.open(gt_path + gt_name)
            gts[i] = np.asarray(g_truth)

            # 对应mask
            masks_name = ""
            if train_test == "train":
                masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test == "test":
                masks_name = files[i][0:2] + "_test_mask.gif"
            b_mask = Image.open(mask_path + masks_name)
            border_masks[i] = np.asarray(b_mask)

    imgs = np.transpose(imgs, (0, 3, 1, 2))  # imgs = np.empty((Nimgs,height,width,channels)) -》 imgs = np.empty((Nimgs,channels，height,width))
    gts = np.reshape(gts, (img_num, 1, height, width))
    border_masks = np.reshape(border_masks, (img_num, 1, height, width))

    return imgs, gts, border_masks


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# 训练集hdf5
imgs_train, groundTruth_train, border_masks_train = get_datasets(imgs_train, groundTruth_train, masks_train, "train")
write_hdf5(imgs_train, dataset_path + "DRIVE_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_groundTruth_train.hdf5")
write_hdf5(border_masks_train, dataset_path + "DRIVE_masks_train.hdf5")

# 测试集hdf5
imgs_test, groundTruth_test, border_masks_test = get_datasets(imgs_test, groundTruth_test, masks_test, "test")
write_hdf5(imgs_test, dataset_path + "DRIVE_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_groundTruth_test.hdf5")
write_hdf5(border_masks_test, dataset_path + "DRIVE_masks_test.hdf5")

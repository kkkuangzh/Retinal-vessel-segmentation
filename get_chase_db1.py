
# 准备CHASE_DB数据集

import os
import h5py
import numpy as np
from PIL import Image


def write_hdf5(arr, outfile):
  with h5py.File(outfile, "w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


# 图片路径
original_imgs = "./CHASEDB1/images/"
groundTruth = "./CHASEDB1/manual/"

img_num = 28
channels = 3
height = 960
width = 999
dataset_path = "./CHASE_DB_dataset/"


def get_datasets(imgs_path, gts_path):
    imgs = np.empty((img_num, height, width, channels))
    gts = np.empty((img_num, height, width))
    for path, subpath, files in os.walk(imgs_path):
        for i in range(len(files)):
            img = Image.open(imgs_path+files[i])
            imgs[i] = np.asarray(img)
            gt_name = files[i][0:2] + "_manual1.png"
            g_truth = Image.open(gts_path + gt_name)
            gts[i] = np.asarray(g_truth)

    imgs = np.transpose(imgs, (0, 3, 1, 2))  # imgs = np.empty((Nimgs,height,width,channels)) -》 imgs = np.empty((Nimgs,channels，height,width))
    gts = np.reshape(gts, (img_num, 1, height, width))
    return imgs, gts


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

imgs, groundTruth = get_datasets(original_imgs, groundTruth)
write_hdf5(imgs, dataset_path + "CHASEDB1_imgs.hdf5")
write_hdf5(groundTruth, dataset_path + "CHASEDB1_groundTruth.hdf5")

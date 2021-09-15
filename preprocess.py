
# 预处理：绿色通道灰度化 -> 标准化 -> CLAHE -> Gamma变化 -> 归一化

import cv2
from help_functions import *


def PreProc(data):
    train_imgs = rgb2gray(data)  # 提取G通道
    train_imgs = normalized(train_imgs)
    train_imgs = CLAHE(train_imgs)
    train_imgs = gamma_trans(train_imgs, 1.2)
    train_imgs = train_imgs/255.  # 归一化
    return train_imgs


def rgb2gray(rgb):
    grayimgs = rgb[:,0,:,:]*0 + rgb[:,1,:,:]*1 + rgb[:,2,:,:]*0
    grayimgs = np.reshape(grayimgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return grayimgs


def normalized(imgs):
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255 #乘以255不是又把0~1变回去了吗
    return imgs_normalized


def CLAHE(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def gamma_trans(imgs, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    res = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        res[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return res

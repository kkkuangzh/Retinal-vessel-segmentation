
# 训练图像填充，随机切片
# 测试图像填充，顺序切片，重叠切片
# 测试图像顺序复原，重叠复原

import numpy as np
import random
import math

from help_functions import load_hdf5
from preprocess import PreProc


def train_patch(DRIVE_train_imgs, DRIVE_train_groudTruth, patch_height, patch_width, N_subimgs):
    train_imgs_original = load_hdf5(DRIVE_train_imgs)
    train_masks = load_hdf5(DRIVE_train_groudTruth)

    train_imgs = PreProc(train_imgs_original)
    train_masks = train_masks/255.

    patches_imgs_train, patches_masks_train = random_patch(train_imgs, train_masks, patch_height, patch_width, N_subimgs)

    return patches_imgs_train, patches_masks_train


def test_patch(test_imgs, test_num, patch_height, patch_width, stride_height, stride_width, avg):
    test_imgs_original = load_hdf5(test_imgs)
    test_imgs = PreProc(test_imgs_original)
    test_imgs = test_imgs[0:test_num, :, :, :]

    if avg == True:
        test_imgs = extend_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
        test_patches = overlap_patch(test_imgs, patch_height, patch_width, stride_height, stride_width)
        return test_patches, test_imgs.shape[2], test_imgs.shape[3]
    else:
        test_imgs = extend(test_imgs, patch_height, patch_width)
        test_patches = ordered_patch(test_imgs, patch_height, patch_width)
        return test_patches, test_imgs.shape[2], test_imgs.shape[3]


# 随机切片
def random_patch(imgs, masks, patch_h, patch_w, patch_num):
    img_num = imgs.shape[0]
    patches_imgs = np.empty((patch_num, imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((patch_num, masks.shape[1], patch_h, patch_w))
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    patch_per_img = int(patch_num/img_num)

    k = 0
    for i in range(img_num):
        j = 0
        while j < patch_per_img:
            # 切片中心点与边界
            x_center = random.randint(0+int(patch_w/2), img_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2), img_h-int(patch_h/2))
            range_h_l = y_center-int(patch_h/2)
            range_h_r = y_center+int(patch_h/2)
            range_w_l = x_center-int(patch_w/2)
            range_w_r = x_center+int(patch_w/2)

            patch = imgs[i, :, range_h_l:range_h_r, range_w_l:range_w_r]
            patch_mask = masks[i, :, range_h_l:range_h_r, range_w_l:range_w_r]

            patches_imgs[k] = patch
            patches_masks[k] = patch_mask

            k += 1
            j += 1

    return patches_imgs, patches_masks


def extend(img, patch_h, patch_w):
    img_h = img.shape[2]
    img_w = img.shape[3]
    new_img_h = 0
    new_img_w = 0

    if (img_h % patch_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = math.ceil(img_h/patch_h)*patch_h
    if (img_w % patch_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = math.ceil(img_w/patch_w)*patch_w

    new_img = np.zeros((img.shape[0], img.shape[1], new_img_h, new_img_w))
    new_img[:, :, 0:img_h, 0:img_w] = img[:, :, :, :]

    return new_img


def extend_overlap(img, patch_h, patch_w, stride_h, stride_w):
    img_h = img.shape[2]
    img_w = img.shape[3]
    remain_h = (img_h-patch_h) % stride_h
    remain_w = (img_w-patch_w) % stride_w
    fill_h = stride_h-remain_h
    fill_w = stride_w-remain_w

    if (remain_h != 0):
        temp = np.zeros((img.shape[0], img.shape[1], img_h + fill_h, img.shape[3]))
        temp[0:img.shape[0], 0:img.shape[1], 0:img_h, 0:img.shape[3]] = img
        img = temp

    if (remain_w != 0):
        temp = np.zeros((img.shape[0], img.shape[1], img.shape[2], img_w + fill_w))
        temp[0:img.shape[0], 0:img.shape[1], 0:img.shape[2], 0:img_w] = img
        img = temp

    return img


# 顺序切片
def ordered_patch(imgs, patch_h, patch_w):
    img_num = imgs.shape[0]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]

    num_h = int(img_h/patch_h)
    num_w = int(img_w/patch_w)
    patch_num = (num_h*num_w)*img_num
    patch_extracted = np.empty((patch_num, imgs.shape[1], patch_h, patch_w))

    k = 0
    for i in range(img_num):
        for h in range(num_h):
            for w in range(num_w):
                patch = imgs[i, :, h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w*patch_w)+patch_w]
                patch_extracted[k] = patch
                k += 1

    return patch_extracted


# 重叠切片
def overlap_patch(imgs, patch_h, patch_w, stride_h, stride_w):
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    img_num = imgs.shape[0]

    num_h = (img_h-patch_h)//stride_h + 1
    num_w = (img_w-patch_w)//stride_w + 1
    patch_num = (num_h*num_w)*img_num
    patch_extracted = np.empty((patch_num, imgs.shape[1], patch_h, patch_w))

    k = 0
    for i in range(img_num):
        for h in range(num_h):
            for w in range(num_w):
                patch = imgs[i, :, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w]
                patch_extracted[k] = patch
                k += 1

    return patch_extracted


# 顺讯切片复原图像
def recompone(patches, img_h, img_w):
    patch_h = patches.shape[2]
    patch_w = patches.shape[3]
    num_h = int(img_h/patch_h)
    num_w = int(img_w/patch_w)
    imgs_num = int(patches.shape[0]/(num_w*num_h))
    predicted_imgs = np.empty((imgs_num, patches.shape[1], num_h*patch_h, num_w*patch_w))
    k = 0
    total = 0

    while total < patches.shape[0]:
        temp_img = np.empty((patches.shape[1], num_h*patch_h, num_w*patch_w))
        for h in range(num_h):
            for w in range(num_w):
                temp_img[:, h*patch_h:(h*patch_h)+patch_h, w*patch_w:(w*patch_w)+patch_w] = patches[total]
                total += 1
        predicted_imgs[k] = temp_img
        k += 1

    return predicted_imgs


# 重叠切复原图像
def recompone_overlap(patches, img_h, img_w, stride_h, stride_w):
    patch_h = patches.shape[2]
    patch_w = patches.shape[3]
    num_h = (img_h-patch_h)//stride_h + 1
    num_w = (img_w-patch_w)//stride_w + 1
    patch_num_each = num_h * num_w
    img_num = patches.shape[0]//patch_num_each

    prob = np.zeros((img_num, patches.shape[1], img_h, img_w))
    sum = np.zeros((img_num, patches.shape[1], img_h, img_w))

    k = 0
    for i in range(img_num):
        for h in range(num_h):
            for w in range(num_w):
                prob[i, :, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w] += patches[k]
                sum[i, :, h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w] += 1
                k += 1

    avg = prob/sum
    return avg


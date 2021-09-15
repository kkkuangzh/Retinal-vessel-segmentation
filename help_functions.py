
# hdf5文件读写
# 预测图像块设置阈值
# 可视化图像
# 可视化切片及专家标注

import h5py
import numpy as np
from PIL import Image


def load_hdf5(file):
    with h5py.File(file, "r") as f:
        return f["image"][()]


def write_hdf5(arr, file):
    with h5py.File(file, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def set_threshold(pred, patch_height, patch_width, mode):
    if mode == "original":
        pred = np.reshape(pred, (pred.shape[0], 1, patch_height, patch_width))
        return pred

    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i, j] >= 0.5:
                    pred[i, j] = 1
                else:
                    pred[i, j] = 0
        pred = np.reshape(pred, (pred.shape[0], 1, patch_height, patch_width))
        return pred


# 可视化预测图像
def test_visualize(data, filename):
    if data.shape[1] == 1:
        data = np.reshape(data, (data.shape[0], data.shape[2], data.shape[3]))
        print('data.shape', data.shape)
    for i in range(data.shape[0]):
        img = np.reshape(data[i], (data.shape[1], data.shape[2]))
        print('img.shape', img.shape)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(filename + '_' + str(i) + '.png')


# 调整mask.shape
def set_mask(masks):
    print(masks.shape)
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], im_h * im_w))
    masks = np.reshape(masks, masks.shape + (1,))
    return masks


# 可视化输入切片
def group_images(patch, row):
    patch = np.transpose(patch, (0, 2, 3, 1))
    temp = []
    for i in range(int(patch.shape[0] / row)):
        stripe = patch[i * row]  # 0,5,10···
        for j in range(i * row + 1, i * row + row):
            stripe = np.concatenate((stripe, patch[j]), axis=1)  # 横向拼接
        temp.append(stripe)

    res = temp[0]
    for i in range(1, len(temp)):
        res = np.concatenate((res, temp[i]), axis=0)  # 纵向拼接
    return res


def visualize(data, filename):
    if data.shape[2] == 1:
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    img = Image.fromarray((data * 255).astype(np.uint8))
    img.save(filename + '.png')
    return img

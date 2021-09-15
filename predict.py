
# 实现顺序切片与重叠切片
# 复原预测结果

import os
from keras.models import model_from_json
from help_functions import  *
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import test_patch


DRIVE_test_imgs_original = './DRIVE_dataset/DRIVE_imgs_test.hdf5'
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]


# 图像块参数
patch_height = 48
patch_width = 48
Imgs_to_test = 5         # 设定测试图像数量

# 结果路径
result_path = './result/predict_result/'

if not os.path.exists(result_path):
    os.makedirs(result_path)

# 重叠切片
average_mode = False    # 选择重叠切片还是顺序切片
stride_height = 20
stride_width = 20


patches_imgs_test, new_height, new_width = \
    test_patch(DRIVE_test_imgs_original, Imgs_to_test, patch_height, patch_width, stride_height, stride_width, average_mode)

# 图像块预测
model = model_from_json(open('./result/test architecture.json').read())
model.load_weights('./result/test_best_weights.h5')
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)

# 设置阈值
pred_patches = set_threshold(predictions, patch_height, patch_width, "original")  # 选择输出原始概率图或阈值结果 original/threshold

if average_mode:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)
else:
    pred_imgs = recompone(pred_patches, new_height, new_width)

pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
test_visualize(pred_imgs, result_path+"prediction")




# 定义并训练网络模型
# 层间拼接FCN与基于稠密连接单元的FCN

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout, BatchNormalization, add
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import os

from help_functions import *
from extract_patches import train_patch
import matplotlib.pyplot as plt


# 层间连接的全卷积神经网络
def network1(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    conv6 = Conv2D(2, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)

    conv6 = Conv2D(1, (1, 1), activation='sigmoid',padding='same',data_format='channels_first')(conv6)
    conv7 = core.Reshape((1,patch_height*patch_width))(conv6)
    conv7 = core.Permute((2,1))(conv7)          # 将第二个维度重排到第一个维度

    model = Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def DenseBlock(inputs, outdim):
    inputshape = K.int_shape(inputs)
    bn = BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(inputs)
    act = core.Activation('relu')(bn)
    conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same', data_format='channels_first')(act)

    if inputshape[1] != outdim:
        shortcut = Conv2D(outdim, (1, 1), padding='same', data_format='channels_first')(inputs)
    else:
        shortcut = inputs
    result1 = add([conv1, shortcut])

    bn = BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(result1)
    act = core.Activation('relu')(bn)
    act = Dropout(0.2)(act)
    conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same', data_format='channels_first')(act)
    result = add([result1, conv2, shortcut])
    result = core.Activation('relu')(result)
    return result


# 基于稠密连接单元的全卷积神经网络
def network2(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (1, 1), activation=None, padding='same', data_format='channels_first')(inputs)
    conv1 = BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv1)
    conv1 = core.Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv1 = DenseBlock(conv1, 32)  # 48
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(pool1, 64)  # 24
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(pool2, 128)  # 12

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([up1, conv2], axis=1)

    conv5 = DenseBlock(up1, 64)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([up2, conv1], axis=1)

    conv6 = DenseBlock(up2, 32)
    conv6 = Conv2D(2, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    conv6 = Conv2D(1, (1, 1), activation='sigmoid', padding='same', data_format='channels_first')(conv6)
    conv7 = core.Reshape((1,patch_height*patch_width))(conv6)
    conv7 = core.Permute((2,1))(conv7)          # 将第二个维度重排到第一个维度

    model = Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# 训练参数设置
N_epochs = 5
batch_size = 32
patch_height = 48
patch_width =48
Total_patch = 20000   # 20幅图像的整数

# 随机提取图像块
patches_imgs_train, patches_masks_train = train_patch('./DRIVE_dataset/DRIVE_imgs_train.hdf5',
                                                      './DRIVE_dataset/DRIVE_groundTruth_train.hdf5',
                                                      patch_height, patch_width, Total_patch,)


# 可视化输入切片与对应专家分割结果
#sample_num = 20
#visualize(group_images(patches_imgs_train[0:sample_num, :, :, :], 5), './result/sample_patches')
#visualize(group_images(patches_masks_train[0:sample_num, :, :, :], 5), './result/sample_patches_masks')


# 保存网络结构，定义并训练模型
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]

# 选择网络结构
model = network1(n_ch, patch_height, patch_width)
#model = network2(n_ch, patch_height, patch_width)

if not os.path.exists('./result'):
    os.makedirs('./result')

json_string = model.to_json()
open('./result/test architecture.json', 'w').write(json_string)

checkpointer = ModelCheckpoint(filepath='./result/test_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
patches_masks_train = set_mask(patches_masks_train)

history = model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])


# 绘制训练曲线
def plot_train(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.plot(epochs, acc, label='train_acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('loss and acc')
    plt.title('Training Performance')
    plt.legend()
    plt.savefig("./result/train.png")
    plt.show()


plot_train(history)

# GPU运行时无法使用checkpointer
# model.save_weights('./result/test_last_weights.h5', overwrite=True)
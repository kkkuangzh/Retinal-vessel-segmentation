# Three components for MesNet
# MFP: Multi-scale Feature Pre-extraction
# ESCP: Encoder Spatial Cascading Path
# SE: Squeeze and Excitation block (attention)

from keras.layers import core, Input, Conv2D, Dense, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Concatenate, Multiply, GlobalMaxPooling2D
from keras.models import Model

def MFP_Block(inputs):
    branch1 = Conv2D(32, 3, padding='same')(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation("relu")(branch1)

    branch2 = Conv2D(32, 3, dilation_rate=3, padding='same')(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation("relu")(branch2)

    branch3 = Conv2D(32, 5, padding='same')(inputs)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation("relu")(branch3)

    branch4 = Conv2D(32, 3, padding='same')(inputs)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation("relu")(branch4)
    branch4 = Conv2D(32, 1, padding='same')(branch4)
    branch4 = BatchNormalization()(branch4)
    branch4 = Activation("relu")(branch4)

    branch5 = Conv2D(32, 3, dilation_rate=3, padding='same')(inputs)
    branch5 = BatchNormalization()(branch5)
    branch5 = Activation("relu")(branch5)
    branch5 = Conv2D(32, 3, dilation_rate=5, padding='same')(branch5)
    branch5 = BatchNormalization()(branch5)
    branch5 = Activation("relu")(branch5)

    merge = Concatenate()([branch1, branch2, branch3, branch4, branch5])
    return merge


def ESCP(input1, input2, input3):
    # downsampling to the same (h*w)
    input1 = MaxPooling2D((2, 2))(input1)
    input1 = MaxPooling2D((2, 2))(input1)
    input2 = MaxPooling2D((2, 2))(input2)

    inputs = Concatenate()([input1, input2, input3])
    conv1 = Conv2D(128, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(64, 3, dilation_rate=3, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)

    conv3 = Conv2D(64, 3, dilation_rate=5, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)

    merge = Concatenate()([conv1, conv2, conv3])
    return merge

# 返回一个概率
def SE_Block(feature_map):
    _, w, h, c = feature_map.get_shape()
    inputs = feature_map
    gp = GlobalMaxPooling2D()(inputs)
    dense1 = Dense(c // 16, activation='relu')(gp)
    dense2 = Dense(c, activation='sigmoid')(dense1)
    # dense2 = core.Reshape((1, 1, c))(dense2)
    return dense2



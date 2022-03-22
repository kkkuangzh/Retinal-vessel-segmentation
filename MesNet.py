from Blocks import *

def MesNet():
    # 参数变化
    # 原始U-net:           Total: 1,962,149, Trainable: 1,959,267, Non-trainable: 2,882
    # 加了3个SE block后:    Total: 1,987,055, Trainable: 1,984,173, Non-trainable: 2,882
    # 加了ESCP后:           Total: 4,152,943, Trainable: 4,148,589  Non-trainable: 4,354
    # 加了MFP后:            Total: 4,237,839, Trainable: 4,233,037, Non-trainable: 4,802

    # encoder
    # Level1 - MFP (512, 512, 3)
    input1 = Input(shape=(512, 512, 3), name="level1")
    MFP1 = MFP_Block(input1)

    conv1 = Conv2D(32, 3, padding='same')(MFP1)
    bn1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    ac1 = Activation('relu')(bn1)
    conv2 = Conv2D(32, 3, padding='same')(ac1)
    bn2 = BatchNormalization()(conv2)
    ac2 = Activation('relu')(bn2)
    pool1 = MaxPooling2D((2, 2))(ac2)

    # Level2 - MFP (256, 256, 3)
    input2 = Input(shape=(256, 256, 3), name="level2")
    pool1 = Concatenate()([pool1, input2])

    conv3 = Conv2D(64, 3, padding='same')(pool1)
    bn3 = BatchNormalization()(conv3)
    ac3 = Activation('relu')(bn3)
    conv4 = Conv2D(64, 3, padding='same')(ac3)
    bn4 = BatchNormalization()(conv4)
    ac4 = Activation('relu')(bn4)
    pool2 = MaxPooling2D((2, 2))(ac4)

    # Level3 - MFP (128, 128, 3)
    input3 = Input(shape=(128, 128, 3), name="level3")
    pool2 = Concatenate()([pool2, input3])

    conv5 = Conv2D(128, 3, padding='same')(pool2)
    bn5 = BatchNormalization()(conv5)
    ac5 = Activation('relu')(bn5)
    conv6 = Conv2D(128, 3, padding='same')(ac5)
    bn6 = BatchNormalization()(conv6)
    ac6 = Activation('relu')(bn6)
    pool3 = MaxPooling2D((2, 2))(ac6)

    # Level4 - MFP (64, 64, 3)
    input4 = Input(shape=(64, 64, 3), name="level4")
    pool3 = Concatenate()([pool3, input4])

    conv7 = Conv2D(256, 3, padding='same')(pool3)
    bn7 = BatchNormalization()(conv7)
    ac7 = Activation('relu')(bn7)
    conv8 = Conv2D(256, 3, padding='same')(ac7)
    bn8 = BatchNormalization()(conv8)
    ac8 = Activation('relu')(bn8)

    # Encoder Spatial Cascading Path
    escp = ESCP(pool1, pool2, pool3)

    # Decoder
    # Level4 - Squeeze and Excitation
    merge1 = Concatenate()([escp, ac8])
    se1 = SE_Block(merge1)
    se1 = Multiply(name='attention1')([merge1, se1])

    conv9 = Conv2D(256, 3, padding='same')(se1)
    bn9 = BatchNormalization()(conv9)
    ac9 = Activation('relu')(bn9)
    conv10 = Conv2D(256, 3, padding='same')(ac9)
    bn10 = BatchNormalization()(conv10)
    ac10 = Activation('relu')(bn10)

    # Level3 - Squeeze and Excitation
    up2 = UpSampling2D((2, 2))(ac10)
    merge2 = Concatenate()([up2, ac6])
    se2 = SE_Block(merge2)
    se2 = Multiply(name='attention2')([merge2, se2])

    conv11 = Conv2D(128, 3, padding='same')(se2)
    bn11 = BatchNormalization()(conv11)
    ac11 = Activation('relu')(bn11)
    conv12 = Conv2D(128, 3, padding='same')(ac11)
    bn12 = BatchNormalization()(conv12)
    ac12 = Activation('relu')(bn12)

    # Level2 - Squeeze and Excitation
    up3 = UpSampling2D((2, 2))(ac12)
    merge3 = Concatenate()([up3, ac4])
    se3 = SE_Block(merge3)
    se3 = Multiply(name='attention3')([merge3, se3])

    conv13 = Conv2D(64, 3, padding='same')(se3)
    bn13 = BatchNormalization()(conv13)
    ac13 = Activation('relu')(bn13)
    conv14 = Conv2D(64, 3, padding='same')(ac13)
    bn14 = BatchNormalization()(conv14)
    ac14 = Activation('relu')(bn14)

    # Level1 - Squeeze and Excitation
    up4 = UpSampling2D((2, 2))(ac14)
    merge4 = Concatenate()([up4, ac2])
    se4 = SE_Block(merge4)
    se4 = Multiply(name='attention4')([merge4, se4])

    # output
    conv15 = Conv2D(32, 3, padding='same')(se4)
    bn15 = BatchNormalization()(conv15)
    ac15 = Activation('relu')(bn15)
    conv16 = Conv2D(32, 3, padding='same')(ac15)
    bn16 = BatchNormalization()(conv16)
    ac16 = Activation('relu')(bn16)
    conv17 = Conv2D(1, 3, padding='same')(ac16)
    bn17 = BatchNormalization()(conv17)
    ac17 = Activation('sigmoid')(bn17)

    model = Model(inputs=[input1, input2, input3, input4], outputs=ac17)
    return model

# inputs = np.array((512, 512, 3))
model = MesNet()
model.summary()
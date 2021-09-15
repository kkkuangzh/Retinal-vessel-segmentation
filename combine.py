
# 将低阈值下中心区域融合进高阈值下预测结果

import cv2
import os

high_path = './test/normal_0.5_result/'
low_path = './test/normal_0.3_result/'


def center_combine(hpath, lpath, num):
    for k in range(num):
        himg = cv2.imread(hpath + 'prediction_' + str(k) + '.png')
        himg = cv2.cvtColor(himg, cv2.COLOR_BGR2GRAY)
        limg = cv2.imread(lpath + 'prediction_' + str(k) + '.png')
        limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        for i in range(194, 388):
            for j in range(188, 376):
                himg[i][j] = limg[i][j]

        if not os.path.exists('./result/combine'):
            os.makedirs('./result/combine')
        cv2.imwrite('./result/combine/' + str(k) + '.png', himg)


center_combine(high_path, low_path, 20)
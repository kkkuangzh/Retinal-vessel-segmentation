
# 计算准确率、特异性、灵敏度、精确率、AUC值

import cv2
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


result_path = './test/result/'
expert_path = './test/expert/'
combine_path = './test/combine/'
path2 = './test/normal_0.5_result/'
avg_path = './test/average_result/'
normal_path = './test/normal_result/'


def get_evaluate(path, num):
    acc = []
    sen = []
    spe = []
    pre = []
    auc = []
    for i in range(num):
        imgs = []
        gts = []
        img = cv2.imread(path + 'prediction_' + str(i) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = cv2.imread(path + 'groundTruth_' + str(i) + '.png')
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                imgs.append(img[j,k])
                gts.append(gt[j,k])
        imgs = np.asarray(imgs)
        gts = np.asarray(gts)

        # AUC值
        AUC_ROC = roc_auc_score(gts, imgs)
        auc.append(AUC_ROC)

        # 混淆矩阵
        threshold_confusion = 0.5
        y_pred = np.empty((imgs.shape[0]))
        for i in range(imgs.shape[0]):
            if imgs[i] >= threshold_confusion:
                y_pred[i] = 255
            else:
                y_pred[i] = 0
        confusion = confusion_matrix(gts, y_pred)
        print(confusion)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
            acc.append(accuracy)
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
            spe.append(specificity)
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
            sen.append(sensitivity)
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
            pre.append(precision)

    return acc, sen, spe, pre, auc


acc, sen, spe, pre, auc = get_evaluate(result_path, 3)


# 保存结果至txt文件

if not os.path.exists('./result/evaluate'):
    os.makedirs('./result/evaluate')

file = open('./evaluate/performances.txt', 'w')
file.write("AUC值 " + str(auc)
                + "\nAcc: " + str(acc)
                + "\nSen: " + str(sen)
                + "\nSpe: " + str(spe)
                + "\nPre: " + str(pre)
                )
file.close()
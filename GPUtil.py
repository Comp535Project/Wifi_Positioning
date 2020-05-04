# -*- coding: utf-8 -*-
# 高斯过程回归，首先要判断，所求的是否满足正太分布，如果满足，就可以用高斯正太回归。可以参考一下代码
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C  # REF就是高斯核函数
from mpl_toolkits.mplot3d import Axes3D  # 实现数据可视化3D
from DataUtil import CleanUtil
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class GPUtil:
    scores = []

    def train(self):
        X_train, X_test, y_train, y_test = CleanUtil('testData.csv').split_train_test(0.3)
        # X_train, X_test, y_train, y_test = CleanUtil('trainingData.csv').split_train_test(0.3)
        # 创建数据集
        # 核函数的取值
        kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
        # 创建高斯过程回归,并训练
        reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        reg.fit(X_train, y_train)  # 这是拟合高斯过程回归的步骤，data[:,:-1]获取前两列元素值，data[:,-1]获取后两列元素的值
        y_pred = reg.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred, normalize=False)
        # self.scores.append(accuracy)
        print(y_test)
        print("/n----------------------------------/n")
        print(y_pred)
        # print(accuracy)

# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets


    # def plot_accuracy(self):
    #     """
    #     plot accuracy graph
    #     :return:
    #     """
    #     plt.plot(self.k_range, self.scores)
    #     plt.xlabel('Value of K')
    #     plt.ylabel('Testing accuracy')
    #     plt.show()

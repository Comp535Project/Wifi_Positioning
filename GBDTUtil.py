import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataUtil import CleanUtil
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing

class GBDTUtil:

    # def __init__(self, filename = 'data/online_data.mat.csv'):
    def __init__(self, filename = 'trainingData.csv'):

        self.scores = []
        self.filename = filename

    def classify(self):
        print("start\n")
        # """
        # classify with knn,loop and find best K
        # :return:None
        # """
        X_train, X_test, y_train, y_test = CleanUtil(self.filename).split_train_test(0.3)
        # data = pd.read_csv(r"./data_train.csv")
        # x_columns = []
        # for x in data.columns:
        #     if x not in ['id', 'label']:
        #         x_columns.append(x)
        # X = data[x_columns]
        # y = data['label']
        # x_train, x_test, y_train, y_test = train_test_split(X, y)
        # 模型训练，使用GBDT算法
        print("!!!!!!!!!!!!!!!!!!!!!!!\n")
        # GradientBoostingClassifier(criterion='friedman_mse', init=None, learning_rate=0.5, loss='deviance', max_depth=3,
        #                            max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
        #                            min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
        #                            min_weight_fraction_leaf=0.0, n_estimators=100, presort='auto', random_state=None,
        #                            subsample=1.0, verbose=0, warm_start=False)
        #
        # gbr = GradientBoostingClassifier()
        gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
        # gbr.fit(X_train, y_train.ravel())
        print("!!!!!!!!!!!!!!!!!!!!!!!\n")
        gbr.fit(X_train, y_train)
        # joblib.dump(gbr, 'train_model_result4.m')  # 保存模型

        # y_gbr = gbr.predict(x_train)
        # y_gbr1 = gbr.predict(x_test)
        # acc_train = gbr.score(x_train, y_train)
        # acc_test = gbr.score(x_test, y_test)
        # print(acc_train)
        # print(acc_test)
        y_pred = gbr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        print("end\n")
    # def plot_accuracy(self):
    #     """
    #     plot accuracy graph
    #     :return:
    #     """
    #     plt.plot(self.k_range, self.scores)
    #     plt.xlabel('Value of K')
    #     plt.ylabel('Testing accuracy')
    #     plt.show()



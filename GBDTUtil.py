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
from sklearn.preprocessing import StandardScaler
from DataUtil import MatUtil
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
class GBDTUtil:



    # def __init__(self, filename = 'data/online_data.mat.csv'):
    def __init__(self):
        self.scores = []
        self.df = MatUtil(r'./data/offline_data_uniform.mat').mat_to_csv_float32()
        # self.df = pd.concat([self.df, MatUtil(r'./data/offline_data_random.mat').mat_to_csv_float32()], axis=0)
        # df.round({'A': 1, 'C': 2})

    def train_test_split(self, ratio):
        x = self.df.loc[:, 'ap1':'ap6']
        # x = pd.concat([x, self.df.label], axis=1)
        y = self.df.loc[:, 'label']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=0)
        # print(x)
        # print("++++++++++++++++++++++++++++++++")
        # print(y)
        # print("++++++++++++++++++++++++++++++++")
        # print(X_train)
        # print("++++++++++++++++++++++++++++++++")
        # print(X_test)
        # print("++++++++++++++++++++++++++++++++")
        # print(y_train)
        # print("++++++++++++++++++++++++++++++++")
        # print(y_test)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    def classify(self):
        # """
        # classify with knn,loop and find best K
        # :return:None
        # """
        X_train, X_test, y_train, y_test = self.train_test_split(0.3)
        # gbr = GradientBoostingClassifier(verbose = 1, n_estimators=100, max_depth=2, min_samples_split=2, learning_rate=0.2)
        # gbr = GradientBoostingClassifier(verbose = 1, random_state=10)
        # gbr.fit(X_train, y_train.ravel())
        # gbr.fit(X_train, y_train)
        param_test1 = {'n_estimators': range(20, 81, 10)}
        gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(verbose = 1,
                                                                     random_state=10),
                                param_grid=param_test1,  iid=False, cv=5)
        # y_train = preprocessing.label_binarize(y_train, classes=[0, 1, 2, 3])
        gsearch1.fit(X_train, y_train)
        gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


        # joblib.dump(gbr, 'train_model_result4.m')  # 保存模型

        # y_gbr = gbr.predict(x_train)
        # y_gbr1 = gbr.predict(x_test)
        # acc_train = gbr.score(x_train, y_train)
        # acc_test = gbr.score(x_test, y_test)
        # print(acc_train)
        # print(acc_test)
        # y_pred = gbr.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # print(accuracy)
        # print("end\n")
    # def plot_accuracy(self):
    #     """
    #     plot accuracy graph
    #     :return:
    #     """
    #     plt.plot(self.k_range, self.scores)
    #     plt.xlabel('Value of K')
    #     plt.ylabel('Testing accuracy')
    #     plt.show()



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataUtil import KaggleDataUtil
from DataUtil import prepare_Mat
import matplotlib.pyplot as plt
import time
import pandas as pd
from ProjectConstant import *
from RandomForestUtil import mergeLabeling
class KNNUtil:

    def __init__(self, k, ratio,based_label):
        self.scores = []
        self.k = k
        self.k_range = range(1, k)
        self.ratio = ratio
        self.basedLabel = based_label


    def train_test_split_knn(self,ratio):
        df = prepare_Mat()
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        df = df.dropna()
        df = mergeLabeling(df,ratio)
        x = df.loc[:, 'ap1':'ap6']
        if self.basedLabel == PREDICT_BY_LABEL:
            x = pd.concat([x, df.rf_label_direct], axis=1)
        elif self.basedLabel == PREDICT_BY_COORDINATE:
            x = pd.concat([x, df.rf_label_coord], axis=1)
        y = df.loc[:, 'x':'y']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=0)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


    def compute_distances_no_loops(self, X_test,X_train):
        """
        Compute Euclidian distance
        :param X_test: test point from ap1 to ap6
        :param X_train: train point from ap1 to ap6
        :return:
        """
        num_test = X_test.shape[0]
        num_train = X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        test_sum = np.sum(np.square(X_test), axis=1)
        train_sum = np.sum(np.square(X_train), axis=1)
        inner_product = np.dot(X_test, X_train.T)
        dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum)
        return dists


    def predict_labels(self, dists, y_train,k=1):
        """
        Calculate the predicted position - as labels
        :param dists: element in dists stands for the distance between a test point to train point
        :param y_train: y label matrix
        :param k: k nearest neighbors
        :return: y_pred: prediction of y
        """
        num_test = dists.shape[0]
        y_pred = np.zeros((num_test, 2))
        for i in range(num_test):

            y_indicies = np.argsort(dists[i, :], axis=0)

            closest_y = y_train[y_indicies[: k]]

            y_pred[i] = np.mean(closest_y, axis=0)
        return y_pred

    def compute_coordinate_dist(self, y_test, y_pred):
        """
        compute the distance between prediction coordinate and real coordinate
        :param y_test: real y test
        :param y_pred: y_pred
        :return:
        """
        num_test = y_test.shape[0]
        coordinate_dist = np.zeros((num_test, 1))
        for i in range(num_test):
            coordinate_dist[i] = np.sqrt(np.sum(np.square(y_test[i, :] - y_pred[i, :])))
        return coordinate_dist

    def classify(self):
        """
        classify with knn,loop and find best K
        :return:None
        """
        X_train, X_test, y_train, y_test = self.train_test_split_knn(self.ratio)
        distance = self.compute_distances_no_loops(X_test,X_train)
        for ki in self.k_range:
            print("k = " + str(ki) + " begin ")
            start = time.time()
            y_pred = self.predict_labels(distance,y_train,ki)
            coordinate_dist = self.compute_coordinate_dist(y_test,y_pred)
            correct_count = np.sum((coordinate_dist < 200) == True)

            total_count = y_test.shape[0]
            accuracy = float(correct_count) / total_count
            # self.scores.append(accuracy)
            s = 0.0
            num = y_test.shape[0]
            for i in range(num):
                s += coordinate_dist[i] * coordinate_dist[i]
            self.scores.append(np.sqrt(s / num) / 100)
            end = time.time()
            print("Complete time: " + str(end - start) + " Secs.")


    def plot_accuracy(self):
        """
        plot accuracy graph
        :return:
        """
        plt.plot(self.k_range, self.scores)
        plt.xlabel('Value of K')
        plt.ylabel('Root-Mean-Square (m)')
        plt.show()

    def plot_result(self):
        """
        plot the classification result from the scratch
        :return:
        """
        X_train, X_test, y_train, y_test = KaggleDataUtil(self.filename).split_train_test(0.4)
        # y_pred = self.KNNModel(self.k, X_train, y_train, X_test)
        # 也画出所有的训练集数据
        print(X_test)
        # plt.scatter(X_test[:,'LONGTITUDE'], X_test[:, 'LATITUDE'], c=y_test)
        # plt.show()


def plot_best_knn(based_label,best_k):
    pass
#
# if __name__ == "__main__":
#     knn = KNNUtil(3)
#     knn.classify()




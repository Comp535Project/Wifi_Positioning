from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataUtil import KaggleDataUtil
from DataUtil import prepare_Mat
from DataUtil import MatUtil
from RandomForestUtil import MatRandomForest
from RandomForestUtil import mergeLabeling
from RandomForestUtil import PREDICT_BY_COORINATE,PREDICT_BY_LABEL
import matplotlib.pyplot as plt
import time
import pandas as pd


class KNNUtil:

    def __init__(self, k):
        self.scores = []
        self.k = k
        self.k_range = range(1, k)


    def train_test_split_knn(self,ratio):
        # df = prepare_Mat();
        df = mergeLabeling('./newdata/offline_data_random.csv')
        df = pd.concat([df, mergeLabeling('./newdata/offline_data_random.csv')], axis=0)
        # df = MatUtil(r'./data/offline_data_uniform.mat').mat_to_csv()
        # df = pd.concat([df, MatUtil(r'./data/offline_data_random.mat').mat_to_csv()], axis=0)
        # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        # df = df.dropna()
        x = df.loc[:, 'ap1':'ap6']
        x = pd.concat([x, df.label], axis=1)
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
        X_train, X_test, y_train, y_test = self.train_test_split_knn(0.6)
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

        # new_df = MatUtil(r'./data/offline_data_uniform.mat').mat_to_csv()
        # new_df = pd.concat([new_df, MatUtil(r'./data/offline_data_random.mat').mat_to_csv()], axis=0)
        new_df = mergeLabeling('./newdata/offline_data_random.csv')
        new_df = pd.concat([new_df, mergeLabeling('./newdata/offline_data_random.csv')], axis=0)
        new_df = new_df[~new_df.isin([np.nan, np.inf, -np.inf]).any(1)]
        new_df = new_df.dropna()
        # new_df = new_df.loc[:, 'ap1':'ap6']
        # X_all = new_df.loc[:, 'ap1':'ap6']
        # X_all = pd.concat([X_all, new_df.label], axis=1)
        # X_all = X_all.to_numpy()
        distance = self.compute_distances_no_loops(X_test, X_train)
        y_pred = self.predict_labels(distance, y_train, 5)
        frame2 = pd.DataFrame(y_pred)
        frame1 = pd.DataFrame(X_test)
        frame1 = pd.concat([frame1, frame2], axis=1)
        frame1.columns = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6','label', 'x', 'y']
        frame1['label'] = (frame1.x // 240) + ((frame1.y // 20) - (frame1.y) // 20 % 10)

        MatUtil().SimpleVisulizeCoord(frame1)

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

#
# if __name__ == "__main__":
#     knn = KNNUtil(3)
#     knn.classify()




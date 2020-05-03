from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataUtil import CleanUtil
import matplotlib.pyplot as plt
import time

"""
cross validation - parameter fitting - data visualization
"""

class KNNUtil:

    def __init__(self, k, filename = 'trainingData.csv'):
        self.scores = []
        self.k = k
        self.k_range = range(1, k)
        self.filename = filename

    def classify(self):
        """
        classify with knn,loop and find best K
        :return:None
        """
        X_train, X_test, y_train, y_test = CleanUtil(self.filename).split_train_test(0.3)
        for ki in self.k_range:
            print("k = " + str(ki) + " begin ")
            start = time.time()
            y_pred = self.KNNModel(ki,X_train,y_train,X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.scores.append(accuracy)
            end = time.time()
            # print(classification_report(y_test, y_pred))
            # print(confusion_matrix(y_test, y_pred))
            print("Complete time: " + str(end - start) + " Secs.")

    def KNNModel(self,ki,X_train,y_train,X_test):
        knn = KNeighborsClassifier(n_neighbors=ki)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return y_pred

    def plot_accuracy(self):
        """
        plot accuracy graph
        :return:
        """
        plt.plot(self.k_range, self.scores)
        plt.xlabel('Value of K')
        plt.ylabel('Testing accuracy')
        plt.show()

    def plot_result(self):
        """
        plot the classification result from the scratch
        :return:
        """
        X_train, X_test, y_train, y_test = CleanUtil(self.filename).split_train_test(0.3)
        # y_pred = self.KNNModel(self.k, X_train, y_train, X_test)
        # 也画出所有的训练集数据
        print(X_test)
        # plt.scatter(X_test[:,'LONGTITUDE'], X_test[:, 'LATITUDE'], c=y_test)
        # plt.show()


# if __name__ == "__main__":
#     knn = KNNUtil(3)
#     knn.plot_result()




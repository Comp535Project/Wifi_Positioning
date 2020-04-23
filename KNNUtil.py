from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DataUtil import CleanUtil
import matplotlib.pyplot as plt
import time


class KNNUtil:
    k = 0
    scores = []
    k_range = None

    def __init__(self, k):
        self.k = k
        self.k_range = range(1, k)

    def classify(self):
        """
        classify with knn,loop and find best K
        :return:None
        """
        X_train, X_test, y_train, y_test = CleanUtil('trainingData.csv').split_train_test(0.3)
        # print(X_train.shape,type(X_train), X_test.shape,y_train.shape,y_test.shape)
        for ki in self.k_range:
            print("k = " + str(ki) + " begin ")
            start = time.time()
            knn = KNeighborsClassifier(n_neighbors=ki)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.scores.append(accuracy)
            end = time.time()
            # print(classification_report(y_test, y_pred))
            # print(confusion_matrix(y_test, y_pred))
            print("Complete time: " + str(end - start) + " Secs.")

    def plot_accuracy(self):
        """
        plot accuracy graph
        :return:
        """
        plt.plot(self.k_range, self.scores)
        plt.xlabel('Value of K')
        plt.ylabel('Testing accuracy')
        plt.show()




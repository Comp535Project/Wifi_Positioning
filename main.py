from KNNUtil import KNNUtil

if __name__ == "__main__":
    knn = KNNUtil(5)
    knn.classify()
    knn.plot_accuracy()
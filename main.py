from KNNUtil import KNNUtil

if __name__ == "__main__":
    k = input("input the max K: ")
    knn = KNNUtil(eval(k))
    knn.classify()
    knn.plot_accuracy()
from KNNUtil import KNNUtil
from ProjectConstant import *


if __name__ == "__main__":
    k = input("input the max K: ")
    knn = KNNUtil(eval(k),ratio=0.5,based_label=PREDICT_BY_COORDINATE)
    knn.classify()
    knn.plot_accuracy()
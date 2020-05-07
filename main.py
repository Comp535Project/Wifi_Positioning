from KNNUtil import KNNUtil
from ProjectConstant import *
import matplotlib.pyplot as plt

def mergePlot_knn(knn_models):
    """
    plot all result rms-k lines in different types of knn models
    :param knn_models: list of models
    :return: void
    """
    labels = []
    for model in knn_models:
        k_ranges = model.k_range
        scores = model.scores
        based_label = model.basedLabel
        plt.plot(k_ranges,scores)
        labels.append(based_label)
        plt.xlabel('x')
        plt.ylabel('Root-Mean-Square (m)')
    plt.legend(labels)
    plt.show()


if __name__ == "__main__":
    k = input("input the max K: ")

    knn_original = KNNUtil(eval(k),ratio=0.3,based_label=ORIGINAL_LABEL, fraction=1,
                           allowVisualize=FORBID_PLOT_PROCESS_GRAPH)
    knn_coordinate = KNNUtil(eval(k),ratio=0.3,based_label=PREDICT_BY_COORDINATE, fraction=11,
                             allowVisualize=FORBID_PLOT_PROCESS_GRAPH)
    knn_direct_label = KNNUtil(eval(k),ratio=0.3,based_label=PREDICT_BY_LABEL, fraction=1,
                               allowVisualize=FORBID_PLOT_PROCESS_GRAPH)

    knn_original.classify()
    knn_coordinate.classify()
    knn_direct_label.classify()

    models = [knn_coordinate,knn_direct_label,knn_original]

    mergePlot_knn(models)
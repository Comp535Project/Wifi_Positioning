import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        # print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        # print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")

    elif train == False:
        pred = clf.predict(X_test)
        print("Test Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        # print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        # print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")



if __name__ == "__main__":
    offline_rss =['ap1','ap2','ap3','ap4','ap5','ap6']
    offline_label = ['label']

    # train data
    dataset1 = pd.read_csv('./newdata/offline_data_uniform.csv')
    dataset1 = dataset1[~dataset1.isin([np.nan, np.inf, -np.inf]).any(1)]
    rss_for_train = np.array(dataset1[offline_rss])
    label_for_train = np.array(dataset1[offline_label])

    # test data
    dataset = pd.read_csv('./newdata/offline_data_random.csv')
    dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]
    rss_for_test = np.array(dataset[offline_rss])
    label_for_test = np.array(dataset[offline_label])



    print(rss_for_test.shape)
    print(rss_for_train.shape)
    print(label_for_test.shape)
    print(label_for_train.shape)

    # #选取使用到的数据
    rss_train = rss_for_train
    label_train = label_for_train

    rss_test = rss_for_test
    label_test = label_for_test





    rand_forest1 = RandomForestClassifier()
    rand_forest1.fit(rss_train, label_train)

    print_score(rand_forest1, rss_train, label_train, rss_test, label_test, train=True)
    print_score(rand_forest1, rss_train, label_train, rss_test, label_test, train=False)

    ## change coefficients by ourselves.
    rand_forest = RandomForestClassifier(n_estimators=400, max_depth=50,
                                         min_samples_split=2, random_state=0)
    rand_forest.fit(rss_train, label_train)

    print_score(rand_forest, rss_train, label_train, rss_test, label_test, train=True)
    print_score(rand_forest, rss_train, label_train, rss_test, label_test, train=False)


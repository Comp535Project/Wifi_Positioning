# -*- coding: utf-8 -*-
# @Time    : 2019/12/1 16:24
# @Author  : Xiaolong Cheng
# @File    : indoor_locating.py
from k_nearest_neighbor import KNearestNeighbor


import scipy.io as sio
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



# 数据集加载
dataset = pd.read_csv('data/offline_data_uniform.csv')
dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]
dataset1 = pd.read_csv('data/offline_data_random.csv')
dataset1 = dataset1[~dataset1.isin([np.nan, np.inf, -np.inf]).any(1)]
print(dataset1.head(5))


#Tagging for RF

#转换格式
offline_rss =['ap1','ap2','ap3','ap4','ap5','ap6']
rss_for_test = np.array(dataset[offline_rss])
rss_for_train = np.array(dataset1[offline_rss])

offline_label = ['label']
label_for_test = np.array(dataset[offline_label])
label_for_train = np.array(dataset1[offline_label])

#选取使用到的数据
rss_train = rss_for_train[:10000] #10000,6
label_train = label_for_train[:10000] #10000,2

rss_test = rss_for_test[:3000] #3000,6
label_test = label_for_train[:3000] #3000,2

def adding_labels_aft_rf(predicted_labels,rss_train_array):
    #Tagging for KNN
    rss_train_array['label'] = predicted_labels
    rss_train_array = np.array(rss_train_array)
    print(rss_train_array)


    offline_rsslabel_test =['ap1','ap2','ap3','ap4','ap5','ap6','label']
    rsslabel_for_test = np.array(dataset1[offline_rsslabel_test])
    rsslabel_test = rsslabel_for_test[:3000]  # 3000,6

    offline_location = ['x','y']
    coordinate_for_train = np.array(dataset1[offline_location])

    coordinate_for_test = dataset[offline_location]
    coordinate_for_test = np.array(coordinate_for_test)
    coordinate_test = coordinate_for_test[:3000]  # 3000,2

    coordinate_train = coordinate_for_train[:10000] #10000,2
    print(coordinate_train)



    return rss_train_array,coordinate_train,rsslabel_test,coordinate_test



def reverse_datafram(array):
    df = pd.DataFrame(data=array, columns=['ap1','ap2','ap3','ap4','ap5','ap6'])
    return df





if __name__ == '__main__':
    #find which cluster this rss_train belongs to.
    ## first train the kmeans model. x: rssi, y: labels
    rand_forest1 = RandomForestClassifier()
    rand_forest1.fit(rss_train, label_train)
    label_pred = list(rand_forest1.predict(rss_train))
    print(label_pred[:5])
    rss_train_array = reverse_datafram(rss_train)
    rsslabel_train, coordinate_train, rsslabel_test, coordinate_test = adding_labels_aft_rf(label_pred,rss_train_array)

    classifier=KNearestNeighbor()
    classifier.train(rsslabel_train,coordinate_train)
    dists=classifier.compute_distances_no_loops(rsslabel_test)#计算欧氏距离
    #选取多组k值，寻找准确率较高的k的取值
    for k in range(1,31):
        y_test_pred = classifier.predict_labels(dists, k)
        # 计算测试集中数据的预测坐标与真实坐标的欧式距离
        coordinate_dist = classifier.compute_coordinate_dist(coordinate_test,y_test_pred)
        #统计预测坐标与真实坐标的平均误差，单位是m
        mean_dis_error = np.mean(coordinate_dist)
        # 统计预测坐标与真实坐标误差小于200CM的样本数量
        num_correct = np.sum((coordinate_dist<200) == True)
        num_test = rsslabel_test.shape[0]
        accuracy = float(num_correct) / num_test
        print ('%d / %d k取%d的预测准确度: %f Distance Error 平均误差在: %e' % (num_correct,num_test,k, accuracy, mean_dis_error))
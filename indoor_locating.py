# -*- coding: utf-8 -*-
# @Time    : 2019/12/1 16:24
# @Author  : Xiaolong Cheng
# @File    : indoor_locating.py

from  k_nearest_neighbor import KNearestNeighbor

import scipy.io as sio
import numpy as np

# 数据集加载
dataset = sio.loadmat('data/offline_data_uniform.mat')
dataset1 = sio.loadmat('data/offline_data_random.mat')
print(list(dataset1))
#转换格式
rss_for_test = np.array(dataset['offline_rss'])
rss_for_train = np.array(dataset1['offline_rss'])
coordinate_for_test = np.array(dataset['offline_location'])
coordinate_for_train = np.array(dataset1['offline_location'])

#选取使用到的数据
rss_train = rss_for_train[:20000] #10000,6
coordinate_train = coordinate_for_train[:20000] #10000,2
rss_test = rss_for_test[:3000] #3000,6
coordinate_test = coordinate_for_test[:3000] #3000,2

if __name__ == '__main__':
    classifier=KNearestNeighbor()
    classifier.train(rss_train,coordinate_train)
    dists=classifier.compute_distances_no_loops(rss_test)#计算欧氏距离
    #选取多组k值，寻找准确率较高的k的取值
    for k in range(1,31):
        y_test_pred = classifier.predict_labels(dists, k)
        # 计算测试集中数据的预测坐标与真实坐标的欧式距离
        coordinate_dist = classifier.compute_coordinate_dist(coordinate_test,y_test_pred)
        #统计预测坐标与真实坐标误差小于200CM的样本数量
        num_correct = np.sum((coordinate_dist<200) == True)
        # print(num_correct)
        num_test = rss_test.shape[0]
        accuracy = float(num_correct) / num_test
        print ('%d / %d k取%d的预测准确度: %f' % (num_correct,num_test,k, accuracy))
import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass
#加载数据
  def train(self, X, y):
    self.X_train = X
    self.y_train = y

#计算欧式距离
  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    test_sum = np.sum(np.square(X), axis=1)
    train_sum = np.sum(np.square(self.X_train), axis=1)
    inner_product = np.dot(X, self.X_train.T)
    dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum)
    return dists

#计算测试集中数据对应的预测坐标结果
  def predict_labels(self, dists, k=1):
    #根据距离矩阵中一个元素（元素表示的是测试点和训练点之间的距离）
    num_test = dists.shape[0]
    y_pred = np.zeros((num_test,2))
    for i in range(num_test):
      #对距离按列方向进行从小到大排序,输出对应的索引
      y_indicies = np.argsort(dists[i, :], axis=0)
      # 找到前k个最近的距离的坐标
      closest_y = self.y_train[y_indicies[: k]]
      #求出k个最邻近坐标的平均值
      y_pred[i] = np.mean(closest_y,axis= 0)
    return y_pred

#计算测试集中数据的预测坐标与真实坐标的欧式距离
  def compute_coordinate_dist(self,coordinate_test,y_pred):
    num_test = coordinate_test.shape[0]
    coordinate_dist = np.zeros((num_test,1))
    for i in range(num_test):
      coordinate_dist[i] = np.sqrt(np.sum(np.square(coordinate_test[i,:]-y_pred[i,:])))
    return coordinate_dist

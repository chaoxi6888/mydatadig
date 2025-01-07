import numpy as np


class KMeans:
    def __init__(self, data, kcluster):
        self.data = data
        self.kcluster = kcluster

    def get_init_centroids(self):
        data_num = self.data.shape[0]  # 数据顺序打散
        indices = np.random.permutation(data_num)  # 取前kcluster个数据
        centroids_select = self.data[indices[:self.kcluster], :]
        return centroids_select

    def train(self, learning_num):
        init_centroids = self.get_init_centroids()
        data_sum = self.data.shape[0]
        centroids = init_centroids.shape[0]
        med_class_result = np.empty((data_sum, 1))
        # 机器学习过程,根据当前质心将数据归类形成簇,更新新的质心,不断迭代
        for train_num in range(learning_num):
            # 当前质心条件下橙的形成过程
            med_class_result = self.process_cluster(init_centroids)  # 更新当前簇条件下的质心
            centroids = self.compute_new_centroids(med_class_result)
            init_centroids = centroids
            # 返回聚类结果,包括质心
        return med_class_result, centroids

    def process_cluster(self, init_centroids):
        data_num = self.data.shape[0]
        # 设置所有数据所在簇的初值
        closest_distance = np.zeros((data_num, 1))
        # 在kcluster个质心中寻拔data中每-条记录所在的簇,用欧氏距离度量
        for data_index in range(data_num):
            # 设置当前记录的欧氏距离初值
            euc_dis = np.zeros((init_centroids.shape[0], 1))
            # 计算当前记录与各质心间的欧氏距离
            for centroid_index in range(init_centroids.shape[0]):
                # 计算当前数据与当前质心属性间的差值
                distance_diff = self.data[data_index, :] - init_centroids[centroid_index, :]
                # 计算欧氏距离
                euc_dis[centroid_index] = np.sum(distance_diff ** 2)
            # 求出数据与kcluster个质心间的最短欧氏距离对应的索引下标,
            # #将数据归为该质心所在簇
            closest_distance[data_index] = np.argmin(euc_dis)  # 返回簇
        return closest_distance

    def compute_new_centroids(self, med_class_result):
        # 质心特征数
        feature_num = self.data.shape[1]
        # 给新的质心赋初值
        new_centroids = np.empty((self.kcluster, feature_num))  # 求各簇平均值
        for cluster_num in range(self.kcluster):
            # 设置当前质心所在簇的索引条件,寻找簇中所有数据c
            closest_ids = med_class_result == cluster_num
            # 计算簇中数据的各列平均值,为当前簇的新质心特征值
            new_centroids[cluster_num] = np.mean(self.data[closest_ids.flatten(), :], axis=0)
        # 返回新的质心
        return new_centroids

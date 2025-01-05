import matplotlib.pyplot as plt
import pandas as pd
from k_means import KMeans
import numpy as np

# 属性列命名
name = ['sepal-length1', 'sepal-width1', 'petal-length', 'petal-width', 'class']
# 加载数据集
dataset = pd.read_csv('iris.csv', names=name, header=0)
# 获取数据行数
dataset_num = dataset.shape[0]
# 选取用于训练的数据,将dataframe类型的数据转换成ndarray数据，
# 为了便于绘图演示,本算法提取了花瓣长度和宽度数据
train_data = dataset.loc[:, 'petal-length': 'petal-width'].values.reshape(dataset_num, 2)
kcluster_num = input('请输入聚类数:')
kcluster = int(kcluster_num)
# 实例化对象
flower = KMeans(train_data, kcluster)
iterator_num = input('请输入迭代次数:')
train_num = int(iterator_num)
# 调用聚类算法
flower_class, centroids = flower.train(train_num)

x_axis = 'petal-length'
y_axis = 'petal-width'
plt.figure(1, figsize=(8, 5))
# 子图1:原始的带有分类结果的数据
plt.subplot(1, 2, 1)
iris_types = ['sctosa', 'versicolor', 'virginica']
plt.title('oringinal data with class')
for iris_type in iris_types:
    plt.scatter(dataset[x_axis][dataset['class'] == iris_type], dataset[y_axis][dataset['class'] == iris_type],
                label=iris_type)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

# 子图2:使用聚类算法得到的分类结果
plt.subplot(1, 2, 2)
plt.title('class from cluster algorithm ')
for kcluster_index, kcluster_value in enumerate(centroids):
    index_condition = (flower_class == kcluster_index).flatten()
    plt.scatter(dataset[x_axis][index_condition], dataset[y_axis][index_condition], label=kcluster_index)
    plt.scatter(kcluster_value[0], kcluster_value[1], color='black', marker='X')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.show()

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random


class Mybayes:
    def __init__(self, data, split_ratio):
        self.data_set = data
        self.ratio = split_ratio

    def split_data(self):
        num_data = len(self.data_set)
        train_size = int(self.ratio * num_data / 100)
        np.random.shuffle(self.data_set)
        train_data = self.data_set[:train_size]
        test_data = self.data_set[train_size:]
        return train_data, test_data

    @staticmethod
    def numbyclass(train_data):
        data_dict = {}
        numbyclass_dict = {}
        for every_data in train_data:
            if every_data[-1] not in data_dict:
                data_dict[every_data[-1]] = []
                numbyclass_dict[every_data[-1]] = 0
            data_dict[every_data[-1]].append(every_data)
            numbyclass_dict[every_data[-1]] += 1
        # print("numbyclass_dict", numbyclass_dict)
        return data_dict, numbyclass_dict

    @staticmethod
    def compute_prioriprobability(train_data, classdict):
        num_samples = len(train_data)
        priori_probinfo = {}
        for every_class, every_value in classdict.items():
            priori_probinfo[every_class] = every_value / float(num_samples)
        return priori_probinfo

    # 计算属性均值
    @staticmethod
    def mean(attribute):
        attr = [float(x) for x in attribute]  # 字符串转数字
        return sum(attr) / float(len(attr))

    # 计算属性方差
    def var(self, attribute):
        attr = [float(x) for x in attribute]
        avg = self.mean(attr)
        if len(attr) == 1:
            var = 0.0001
        else:
            var = sum([math.pow((x - avg), 2) for x in attr]) / float(len(attr) - 1)
            if var == 0:
                var = 0.0001
        return var

    def compute_meanvarbyattribute(self, datavalue):
        dataset = np.delete(datavalue, -1, axis=1)  # delete label
        sta_meanandvar = [(self.mean(attr), self.var(attr)) for attr in zip(*dataset)]
        return sta_meanandvar

    # 高斯概率密度函数@staticmethod
    def calculateprob(self, x, mean, var):
        exponent = math.exp(math.pow((x - mean), 2) / (-2 * var))
        p = (1 / math.sqrt(2 * math.pi * var)) * exponent
        return p

    def compute_likelihoodprob(self, data_dict, test_data):
        # 分类别条件下属性的可能性
        likelihoodprob = {}
        # 各类别条件下的属性均值与方差
        statistics_by_class = {}
        # 分别计算不同类别条件下的均值和方差
        for classvalue, datavalue in data_dict.items():
            statistics_by_class[classvalue] = self.compute_meanvarbyattribute(datavalue)
        for class_value, statis in statistics_by_class.items():
            likelihoodprob[class_value] = 1
            # 计算该类别条件下各属性的可能性p,通过累乘得到该类别下的可能性
            for i in range(len(statis)):
                mean, var = statis[i]
                x = test_data[i]
                p = self.calculateprob(x, mean, var)
                if p == 0:
                    p = 0.0001
                likelihoodprob[class_value] *= p
        return likelihoodprob

    def bayesianclassifier(self, input_data):
        traindata, testdata = self.split_data()
        datadict, classdict = self.numbyclass(traindata)
        priori_prob = self.compute_prioriprobability(traindata, classdict)
        likelihoodprob = self.compute_likelihoodprob(datadict, input_data)
        # 各类别条件下的后验概率
        result = {}
        for class_value, class_prob in likelihoodprob.items():
            p = class_prob * priori_prob[class_value]
            result[class_value] = p
        return max(result, key=result.get)

    def valueatecorrect(self):
        correct = 0
        dataset = self.data_set
        for every_data in dataset:
            input_data = every_data[:-1]
            classlabel = every_data[-1]
            result = self.bayesianclassifier(input_data)
            if result == classlabel:
                correct += 1
        return correct / len(dataset)


# 导入数据
name = ['sepal-lengthl', 'sepal-widthl', 'petal-length', ' petal-width', 'class']
data_resource = pd.read_csv("iris.csv", names=name, header=0)
dataresource = np.array(data_resource).tolist()
split_ratio = input("请输入参与训练数据集比例,只需输入百分比前的整数值即可:")
ratio = int(split_ratio)
# 生成类对象
flower = Mybayes(dataresource, ratio)
# 选择预测数据
k = random.randint(0, len(dataresource))
con_vector = dataresource[k]
true_data = con_vector[:]
input_data = con_vector[:-1]
print("给定的数据是", input_data, "其真实类别是", true_data[-1])
classprob = flower.bayesianclassifier(input_data)
print("预测的类别是:", classprob)
# 评估分类器的预测准确率
accuary = flower.valueatecorrect()
print("该分类器的准确率为:", (accuary * 100), "%")

# 确保类别名称一致
iris_types = ['setosa', 'versicolor', 'virginica']

# 定义 data_plot 为 dataresource
data_plot = dataresource
x_axis = 'petal-length'
y_axis = ' petal-width'  # Note the leading space
plt.figure(1, figsize=(8, 5))

# Subplot 1: Original data with class labels
plt.subplot(1, 2, 1)
plt.title('Original data with class')
for iris_type in iris_types:
    plt.scatter(data_resource[x_axis][data_resource['class'] == iris_type],
                data_resource[y_axis][data_resource['class'] == iris_type], label=iris_type)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

# Subplot 2: Classification results from Bayes algorithm
plt.subplot(1, 2, 2)
plt.title('Class from Bayes algorithm')
class_data = np.empty((len(data_plot), 1), dtype=int)
for i in range(len(data_plot)):
    con_vector = data_plot[i]
    input_data = con_vector[:-1]
    classprob = flower.bayesianclassifier(input_data)
    class_data[i] = iris_types.index(classprob)
for iris_type in iris_types:
    index_condition = (class_data == iris_types.index(iris_type)).flatten()
    testx = data_resource[x_axis][index_condition]
    testy = data_resource[y_axis][index_condition]
    plt.scatter(testx, testy, label=iris_type)
plt.ylabel(y_axis)
plt.legend()
plt.show()

import pandas as pd  # 导入pandas库
import operator  # 导入operator库
import math  # 导入math库
import pydotplus
from graphviz import Digraph
from IPython.display import Image


def loadDataSet():  # 定义加载数据集的函数
    file_name = r'整理后的生育意愿统计表.xlsx'  # 文件名
    data = pd.read_excel(file_name, sheet_name='Sheet1')  # 读取Excel文件中的Sheet1
    data = data.values.tolist()  # 将数据转换为列表
    return data  # 返回数据


def is_leaves(dataset):  # 定义判断是否为叶节点的函数
    class_list = [row[-1] for row in dataset]  # 获取数据集中的类别列表
    class_num = set(class_list)  # 获取类别的集合
    if len(class_num) == 1:  # 如果类别只有一个
        return 1  # 返回1
    elif len(dataset[0]) == 2:  # 如果数据集只有两个特征
        return 2  # 返回2
    else:  # 否则
        return 0  # 返回0


def majorityVote(class_list):  # 定义多数投票函数
    class_count = {}  # 初始化类别计数字典
    for vote in class_list:  # 遍历类别列表
        if vote[-1] not in class_count:  # 如果类别不在字典中
            class_count[vote[-1]] = vote[0]  # 将类别添加到字典中
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # 按照类别计数排序
    class_num_majority = sorted_class_count[0][0]  # 获取计数最多的类别
    return class_num_majority  # 返回计数最多的类别


def calcEntropy(data_set):  # 定义计算熵的函数
    num_data = 0  # 初始化数据个数
    labels_count = {}  # 初始化标签计数字典
    for data in data_set:  # 遍历数据集
        if data[-1] not in labels_count.keys():  # 如果标签不在字典中
            labels_count[data[-1]] = data[0]  # 将标签添加到字典中
        else:  # 否则
            labels_count[data[-1]] += data[0]  # 增加标签计数
        num_data += data[0]  # 增加数据个数
    entropy = 0.0  # 初始化熵
    for label in labels_count.keys():  # 遍历标签字典
        prob = (float(labels_count[label]) / num_data)  # 计算标签的概率
        entropy -= prob * math.log(prob, 2)  # 计算熵
    return entropy, num_data  # 返回熵和数据个数


def getsubdataset(dataset, feature, value):  # 定义获取子数据集的函数
    subdata = []  # 初始化子数据集
    subdata_num = 0  # 初始化子数据集个数
    for row in dataset:  # 遍历数据集
        if row[feature] == value:  # 如果特征值等于给定值
            subdata_num += row[0]  # 增加子数据集个数
            subdata.append(row)  # 将行添加到子数据集中
    return subdata_num, subdata  # 返回子数据集个数和子数据集


def getPrioFeature(dataset):  # 定义获取最优特征的函数
    Entropy_dataset = calcEntropy(dataset)  # 计算数据集的熵
    largest_gain = 0.0  # 初始化最大增益
    prior_feature_information = []  # 初始化最优特征信息
    prior_feature = 0  # 初始化最优特征
    for feature in range(1, len(dataset[0]) - 1):  # 遍历特征
        feature_data = [row[feature] for row in dataset]  # 获取特征数据
        unique_data = set(feature_data)  # 获取唯一特征值
        entropy_featurevalue_sum = 0  # 初始化特征值熵和
        current_feature_information = []  # 初始化当前特征信息
        for value in unique_data:  # 遍历唯一特征值
            featurevalue_num, featurevalue_dataset = getsubdataset(dataset, feature, value)  # 获取子数据集
            weight = float(featurevalue_num) / Entropy_dataset[1]  # 计算权重
            entropy_featurevalue = calcEntropy(featurevalue_dataset)  # 计算子数据集的熵
            entropy_featurevalue_sum += weight * entropy_featurevalue[0]  # 计算加权熵和
            current_feature_information.append([value, featurevalue_num, featurevalue_dataset])  # 添加当前特征信息
        Gain = Entropy_dataset[0] - entropy_featurevalue_sum  # 计算增益
        if Gain >= largest_gain:  # 如果增益大于等于最大增益
            largest_gain = Gain  # 更新最大增益
            prior_feature_information = current_feature_information  # 更新最优特征信息
            prior_feature = feature - 1  # 更新最优特征
    return prior_feature, prior_feature_information  # 返回最优特征和最优特征信息


def buildDecisionTree(dataset, labels1):  # 定义构建决策树的函数
    if is_leaves(dataset) == 1:  # 如果是叶节点
        return dataset[0][-1]  # 返回类别
    elif is_leaves(dataset) == 2:  # 如果数据集只有两个特征
        class_list = [row[-1] for row in dataset]  # 获取类别列表
        return majorityVote(class_list)  # 返回多数投票结果
    prior_feature, prior_feature_information = getPrioFeature(dataset)  # 获取最优特征和最优特征信息
    m = prior_feature + 1  # 计算特征索引
    labels_value = labels1[prior_feature]  # 获取特征标签
    tree = {labels_value: {}}  # 初始化树
    del labels1[prior_feature]  # 删除特征标签
    for i in range(0, len(prior_feature_information)):  # 遍历最优特征信息
        curr_labels = labels1[:]  # 复制标签
        name = prior_feature_information[i][0]  # 获取特征值
        feature_data = prior_feature_information[i][2]  # 获取特征数据
        feature_value_data = []  # 初始化特征值数据
        for data in feature_data:  # 遍历特征数据
            datal = data[:m] + data[m + 1:]  # 删除特征列
            feature_value_data.append(datal)  # 添加到特征值数据
        tree[labels_value][name] = buildDecisionTree(feature_value_data, curr_labels)  # 递归构建决策树
    return tree  # 返回决策树


def visualize_tree(tree, feature_names):
    def add_nodes_edges(tree, dot=None):
        if dot is None:
            dot = Digraph()
            dot.attr(fontname="SimHei")  # 指定支持中文的字体
            dot.node(name=str(id(tree)), label=next(iter(tree)))
        for k, v in tree.items():
            if isinstance(v, dict):
                dot.node(name=str(id(v)), label=next(iter(v)))
                dot.edge(str(id(tree)), str(id(v)), label=str(k))
                add_nodes_edges(v, dot=dot)
            else:
                dot.node(name=str(id(v)), label=str(v))
                dot.edge(str(id(tree)), str(id(v)), label=str(k))
        return dot

    dot = add_nodes_edges(tree)
    return dot


if __name__ == '__main__':  # 主函数
    data_set = loadDataSet()  # 加载数据集
    labels = ["收入压力", "有足够时间照顾", "不影响职业发展"]  # 定义标签
    D_tree = buildDecisionTree(data_set, labels)  # 构建决策树
    dot = visualize_tree(D_tree, labels)
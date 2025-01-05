import pandas as pd
import operator
import math


def loadDataSet():
    file_name = r'dataset.xlsx'
    data = pd.read_excel(file_name, sheet_name='Sheet1')
    data = data.values.tolist()
    return data


def is_leaves(dataset):
    class_list = [row[-1] for row in dataset]
    class_num = set(class_list)
    if len(class_num) == 1:
        return 1
    elif len(dataset[0]) == 2:
        return 2
    else:
        return 0


def majorityVote(class_list):
    class_count = {}
    for vote in class_list:
        if vote[-1] not in class_count:
            class_count[vote[-1]] = vote[0]
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    class_num_majority = sorted_class_count[0][0]
    return class_num_majority


def calcEntropy(data_set):
    num_data = 0
    labels_count = {}
    # 当前数据集的数据个数
    for data in data_set:
        if data[-1] not in labels_count.keys():
            labels_count[data[-1]] = data[0]
        else:
            labels_count[data[-1]] += data[0]
        num_data += data[0]
    entropy = 0.0
    for label in labels_count.keys():
        prob = (float(labels_count[label]) / num_data)
        entropy = prob * math.log(prob, 2)
    return entropy, num_data


def getsubdataset(dataset, feature, value):
    subdata = []
    subdata_num = 0
    for row in dataset:
        if row[feature] == value:
            subdata_num += row[0]
            subdata.append(row)
    return subdata_num, subdata


def getPrioFeature(dataset):
    Entropy_dataset = calcEntropy(dataset)
    largest_gain = 0.0
    # 最优属性的相关信息
    prior_feature_information = []
    prior_feature = 0
    for feature in range(1, len(dataset[0]) - 1):
        feature_data = [row[feature] for row in dataset]
        unique_data = set(feature_data)
        entropy_featurevalue_sum = 0
        current_feature_information = []
        for value in unique_data:
            featurevalue_num, featurevalue_dataset = getsubdataset(dataset, feature, value)
            weight = float(featurevalue_num) / Entropy_dataset[1]
            entropy_featurevalue = calcEntropy(featurevalue_dataset)
            entropy_featurevalue_sum += weight * entropy_featurevalue[0]
            current_feature_information.append([value, featurevalue_num, featurevalue_dataset])
        Gain = Entropy_dataset[0] - entropy_featurevalue_sum
        if Gain >= largest_gain:
            largest_gain = Gain
            prior_feature_information = current_feature_information
            prior_feature = feature - 1
    return prior_feature, prior_feature_information


def buildDecisionTree(dataset, labels):
    if is_leaves(dataset) == 1:
        return dataset[0][-1]
    elif is_leaves(dataset) == 2:
        class_list = [row[-1] for row in dataset]
        return majorityVote(class_list)
    prior_feature, prior_feature_information = getPrioFeature(dataset)
    m = prior_feature + 1
    labels_value = labels[prior_feature]
    tree = {labels_value: {}}
    del labels[prior_feature]
    for i in range(0, len(prior_feature_information)):
        curr_labels = labels[:]
        name = prior_feature_information[i][0]
        feature_data = prior_feature_information[i][2]
        feature_value_data = []
        for data in feature_data:
            data1 = data[:m] + data[m + 1:]
            feature_value_data.append(data1)
        tree[labels_value][name] = buildDecisionTree(feature_value_data, curr_labels)
    return tree


if __name__ == '__main__':
    data_set = loadDataSet()
    labels = ["年龄", "收入", "学生", "信誉"]
    D_tree = buildDecisionTree(data_set, labels)
    print(D_tree)

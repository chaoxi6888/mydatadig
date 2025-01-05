# 装载事务集数据
def load_data():
    trans_data_set = [['I1', 'I2', 'I5'], ['I2', 'I4'], ['I2', 'I3'], ['I1', 'I2', 'I4'], ['I1', 'I2', 'I4'],
                      ['I1', 'I2', 'I4'], ['I1', 'I2', 'I4'], ['I1', 'I3'], ['I2', 'I3'], ['I1', 'I3'],
                      ['I1', 'I2', 'I3', 'I5'], ['I1', 'I2', 'I3']]
    return trans_data_set


# 获取事务集合函数
def create_itemset(data):
    itemset = []
    for t in data:
        for item in t:
            if [item] not in itemset:
                itemset.append([item])
    return list(map(frozenset, itemset))


def cal_frequency(data, ck):
    can_item_freq = {}
    for t in data:
        for item in ck:
            if item.issubset(t):
                if item in can_item_freq:
                    can_item_freq[item] += 1
                else:
                    can_item_freq[item] = 1
    return can_item_freq


def generate_Lk_by_ck(min_support, support_data, k):
    lk = []
    fre_supportdata = {}
    for item_count in support_data:
        if support_data[item_count] > min_support:
            lk.append(item_count)
            fre_supportdata[item_count] = support_data[item_count]
    return lk, fre_supportdata


def create_ck(lk, item_set, min_support, support_data):
    ck = []
    flag = 0
    for lk_item in lk:
        for item in item_set:
            mm = lk_item
            if item in support_data:
                if support_data[item] > min_support:
                    mm = mm | item
                    if mm not in ck:
                        ck.append(mm)
                        flag = 1
    return ck, flag


def cal_confidence(fre_itemset, dis_itemset, support_data, min_conf):
    rule_conf = ()
    conf = support_data[fre_itemset] / support_data[dis_itemset]
    if conf > min_conf:
        print(dis_itemset, '-->', fre_itemset - dis_itemset, 'conf:', conf)
        rule_conf = (dis_itemset, fre_itemset - dis_itemset, conf)
    return rule_conf


def can_rules(fre_set1, min_conf):
    rules = []
    print('满足最小置信度的所有公式如下：')
    for ii in range(0, len(fre_set1)):
        l1 = fre_set1[ii]
        for j in range(0, len(fre_set1)):
            l2 = fre_set1[j]
            if not (l1.issubset(l2) or l2.issubset(l1) or l1 & l2):
                l = l1 | l2
                if l in fre_set1:
                    t1 = cal_confidence(l, l1, fre_support_data, min_conf)
                    if t1:
                        rules.append(t1)
    return rules


def calc_lift_fitness(rule_data, support_data):
    lift = []
    for item in rule_data:
        lift_value = (item[2] / support_data[item[1]])*10
        lift.append((item[0], item[1], lift_value))
    return lift


# 准备工作
source_data = load_data()
all_itemset = create_itemset(source_data)

# 找出事务集中的所有频繁集
ready_set = all_itemset  # 候选1项集
support = input('请输入最小支持度:')
min_sup = int(support)
for i in range(1, len(all_itemset) - 1):  # 找出事务集中所有的频繁集
    itemcount = cal_frequency(source_data, ready_set)  # 计算候选频繁i项集的支持度
    # 创建频繁i项集fre_set,i最大频繁项集数,fre_support_data为带支持度的频繁i项集
    fre_set, fre_support_data = generate_Lk_by_ck(min_sup, itemcount, i)
    #
    ready_set, flag1 = create_ck(fre_set, all_itemset, min_sup, fre_support_data)
    if not flag1:
        break

#
conf_degree = input('请输入置信度：')
minconf = float(conf_degree)
mid_rules = can_rules(fre_set, minconf)
#
sorted_by_conf = sorted(mid_rules, key=lambda r: r[2], reverse=True)
print('生成的所有规则如下:')
print(sorted_by_conf)
#
lift = []
result = calc_lift_fitness(mid_rules, fre_support_data)
sorted_by_lift_fitness = sorted(result, key=lambda x: x[2], reverse=True)
print('带提升度的公式列表如下:')
print(result)

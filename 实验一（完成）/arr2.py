import numpy as np

# 使用logspace()来创建一个对数等比数组
list1 = np.logspace(0, 9, 10, base=2)
print("对数等比数组：\n", list1)

# 使用zeros()来创建一个0数组
# 创建一个长度为5的，初始值都为0的数组
list2 = np.zeros((3, 2))
print("0数组：\n", list2)

# 使用ones()来创建一个全1数组
list3 = np.ones((4, 4))
print("全一数组：\n", list3)

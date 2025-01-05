import numpy as np

# 使用列表来创建一个numpy数组
list1 = np.array([1, 2, 3, 4, 5])

# 使用range来创建一个numpy数组
list2 = np.array(range(1, 6))

# 使用arrange来创建一个numpy数组
list3 = np.arange(1, 6)

# 使用linspace来创建一个numpy数组
list4 = np.linspace(1, 5, 5)

# 使用元组来创建一个numpy数组
np_array = np.array([list1, list2, list3, list4])
print(np_array)

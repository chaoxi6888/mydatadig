import numpy as np

list1 = np.random.rand(10)  # 创建一维随机数组，服从均匀分布
list2 = np.random.randn(2, 5)  # 创建二维随机数组，服从正态分布
print("一维均匀分布随机数组：\n", list1)
print("二维正态分布随机数组：\n", list2)

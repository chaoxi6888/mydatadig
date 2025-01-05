import numpy as np

# 创建两个数组
array1 = np.array([1.0, 2.0, 3.0])
array2 = np.array([1.0, 2.0, 3.0 + 1e-5])  # 两个数组的最后一个元素略有不同

# 使用 allclose() 检查两个数组是否足够接近
result = np.allclose(array1, array2, atol=1e-4)  # atol 是绝对公差
print(result)

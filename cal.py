import numpy as np
from scipy.linalg import expm

# 定义一个矩阵（numpy 数组）
A = np.array([[1, 2], [3, 4]])

# 用 scipy 计算矩阵指数
expA = expm(A)  # 直接返回 numpy 数组

# 检查类型
print(type(expA))  # 输出: <class 'numpy.ndarray'>

# 可以直接进行 numpy 计算
result = expA @ expA  # 矩阵乘法（@ 是 numpy 的矩阵乘法运算符）
print(result)
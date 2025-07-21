import numpy as np


def print_matrix_powers(A, max_power):
    """
    打印矩阵 A 的幂次结构，从 A^1 到 A^max_power。

    参数:
    A -- 输入矩阵
    max_power -- 最大幂次
    """
    print(f"矩阵 A 的维度: {A.shape}")
    for i in range(1, max_power + 1):
        power_matrix = np.linalg.matrix_power(A, i)
        print(f"A^{i} =")
        print(power_matrix)
        print()


# 定义系统参数
K = 0.8  # 增益
T = 20  # 时间常数
Ts = 1  # 采样时间
D = 15  # 延迟

# 计算离散化系数
a = np.exp(-Ts / T)
b0 = K * T * (1 - a)

# 构建状态矩阵 A
n = D + 1  # 状态向量的维度
A = np.zeros((n, n))
A[0, 0] = a
A[0, -1] = b0
for i in range(1, n):
    A[i, i - 1] = 1

# 打印矩阵 A 的幂次结构
print_matrix_powers(A, 10)
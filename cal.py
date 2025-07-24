import numpy as np

# 假设 D = 15（根据你的代码）
D = 15
n = D + 2  # n = 17
A_model = np.zeros((n, n))  # 初始化全零矩阵

# 假设 A_d_model 是之前计算得到的离散化矩阵（2x2）
# 这里用一个示例矩阵代替（实际替换为你的 A_d_model）
A_d_model = np.array([[1.0, 0.5], [0.3, 0.8]])  # 替换为你的 A_d_model

# 将 A_d_model 填充到 A_model 的左上角
A_model[:2, :2] = A_d_model  # 前两行和前两列

print("A_model 的左上角 2x2 子矩阵:")
print(A_model[:2, :2])  # 验证填充结果
print("\n完整的 A_model 矩阵:")
print(A_model)  # 查看完整矩阵
import numpy as np
import control as ctrl

# 定义系统参数
A = np.array([[-1, 1], [1, -2]])    # 2x2 状态矩阵
B = np.array([[1, -2], [3, 1]])      # 2x2 输入矩阵
C = np.array([[1, 0], [0, 1]])      # 2x2 输出矩阵
D = np.array([[0, 0], [0, 0]])      # 2x2 传递矩阵

# 创建一个连续时间的MIMO状态空间系统
sys_ss = ctrl.ss(A, B, C, D)

# 设置采样时间
T_s = 1  # 采样时间为1秒

# 离散化MIMO系统
sys_d = ctrl.c2d(sys_ss, T_s, method='zoh')  # 使用零阶保持法

# 获取离散化后的状态空间矩阵
A_d, B_d, C_d, D_d = sys_d.A, sys_d.B, sys_d.C, sys_d.D

print("离散化后的A_d矩阵：\n", A_d)
print("离散化后的B_d矩阵：\n", B_d)
print("离散化后的C_d矩阵：\n", C_d)
print("离散化后的D_d矩阵：\n", D_d)

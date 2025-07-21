import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 定义系统参数
K_model = 0.8  # 模型中的增益
T_model = 20   # 模型中的时间常数
K_actual = 0.85  # 实际系统中的增益
T_actual = 22    # 实际系统中的时间常数
Ts = 1           # 采样时间
D = 15           # 延迟
P = 50           # 预测步数
M = 10            # 控制步数
total_steps = 200  # 总仿真时间步
external_disturbances = 0.1 # 外部干扰
measurement_noise = 0.05 # 测量噪声



# 定义约束条件
u_min = -1.0
u_max = 1.0





# 计算离散化系数
a_model = np.exp(-Ts / T_model)
b0_model = K_model * T_model * (1 - a_model)
a_actual = np.exp(-Ts / T_actual)
b0_actual = K_actual * T_actual * (1 - a_actual)

# 构建状态矩阵 A、输入矩阵 b 和输出矩阵 c
n = D + 1  # 状态向量的维度
A_model = np.zeros((n, n))
A_model[0, 0] = a_model
A_model[0, -1] = b0_model
for i in range(1, n):
    A_model[i, i - 1] = 1

A_actual = np.zeros((n, n))
A_actual[0, 0] = a_actual
A_actual[0, -1] = b0_actual
for i in range(1, n):
    A_actual[i, i - 1] = 1

b = np.zeros((n, 1))
b[1, 0] = 1

c = np.zeros((1, n))
c[0, 0] = 1

# 计算 Fx 和 Gx
Fx = np.zeros((n * P, n))
Gx = np.zeros((n * P, M))
for i in range(P):
    Fx[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A_model, i + 1)
    for j in range(min(i + 1, M)):
        Gx[i * n:(i + 1) * n, j] = (np.linalg.matrix_power(A_model, i - j) @ b).flatten()
    # 如果 i >= M，计算累加项
    if i >= M:
        Gx[i * n:(i + 1) * n, M - 1] = sum((np.linalg.matrix_power(A_model, j) @ b).flatten() for j in range(i - M + 1))

# 计算 Fy 和 Gy
Fy = np.zeros((P, n))  # 输出预测矩阵 Fy
Gy = np.zeros((P, M))  # 输入预测矩阵 Gy

for i in range(P):
    Fy[i, :] = c @ np.linalg.matrix_power(A_model, i + 1)

for i in range(P):
    for j in range(min(i + 1, M)):
        Gy[i, j] = (c @ np.linalg.matrix_power(A_model, i - j) @ b).item()
    # 如果 i >= M，计算累加项
    if i >= M:
        Gy[i, M - 1] = sum((c @ np.linalg.matrix_power(A_model, j) @ b).item() for j in range(i - M + 1))

# 初始化状态
x_model = np.zeros(n)  # 模型中的初始状态
x_actual = np.zeros(n)  # 实际系统中的初始状态

# 定义参考轨迹
r = np.ones(P) * 1.0  # 希望输出稳定在 1.0

# 初始化存储结果的数组
y_actual_history = np.zeros(total_steps)  # 实际系统的输出历史
u_history = np.zeros(total_steps)  # 控制输入历史

# 仿真循环
for k in range(total_steps):
    # 定义优化问题
    model = gp.Model("MPC")

    # 定义变量
    U = model.addMVar((M,), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="U")

    # 定义目标函数
    Y_pred =  Fy @ x_model +  Gy @ U
    cost = (Y_pred - r).T @ (Y_pred - r) + U.T @ U + (U[1:] - U[:-1]).T @ (U[1:] - U[:-1])
    model.setObjective(cost, GRB.MINIMIZE)


    model.addConstr(U >= u_min, name="u_min")
    model.addConstr(U <= u_max, name="u_max")

    # 求解优化问题
    model.optimize()

    # 获取最优控制输入
    u_optimal = U.X

    # 应用第一个控制输入到实际系统
    u = u_optimal[0]
    x_actual = A_actual @ x_actual + b * u
    y_actual = c @ x_actual

    # 添加外部干扰和测量噪声
    y_actual += np.random.normal(0, external_disturbances)  # 外部干扰
    y_actual += np.random.normal(0, measurement_noise)  # 测量噪声

    # 保存结果
    y_actual_history[k] = y_actual
    u_history[k] = u

    # 更新模型状态
    x_model = x_actual

# 可视化结果
time = np.arange(total_steps)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, y_actual_history, label='Actual Output')
plt.plot(time, np.ones(total_steps) * 1.0, label='Reference', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('System Output vs Reference')

plt.subplot(2, 1, 2)
plt.plot(time, u_history, label='Control Input')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Input')

plt.tight_layout()
plt.show()
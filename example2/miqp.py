import numpy as np
from scipy.linalg import expm
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 定义系统参数
a_1_model = 0.10
a_2_model = 0.08
b_1_model = 1.5
b_2_model = 1.2
c_1_model = 0.5
c_2_model = 0.4

a_1_actual = 0.102
a_2_actual = 0.082
b_1_actual = 1.52
b_2_actual = 1.22
c_1_actual = 0.51
c_2_actual = 0.41

Ts = 1
D = 15
P = 50
M = 10
total_steps = 100
external_disturbances = 0.01

# 约束
u1_min = -2.0
u1_max = 2.0
u2_min = -2.0
u2_max = 2.0
delta_u1_min = -0.5
delta_u1_max = 0.5
delta_u2_min = -0.5
delta_u2_max = 0.5
positive_delta_u1_constrain = 0.2
negative_delta_u1_constrain = -0.2
positive_delta_u2_constrain = 0.2
negative_delta_u2_constrain = -0.2


# 定义代价函数的权重系数
Q1 = np.diag(np.ones(P) * 1.0)  # 对应 err 的权重矩阵
Q2 = np.diag(np.ones(P) * 1.0)  # 对应 err 的权重矩阵
R_delta1 = np.diag(np.ones(M) * 0.001)  # 对应 U[i+1] - U[i] 的权重矩阵
R_delta2 = np.diag(np.ones(M) * 0.001)  # 对应 U[i+1] - U[i] 的权重矩阵


# 控制目标
r_1 = np.ones(total_steps + P) * 1.0
r_2 = np.ones(total_steps + P) * 1.0

#连续方程系数
A_0_model = np.zeros((2,2))
A_0_model[0,0] = -a_1_model
A_0_model[0,1] = 0
A_0_model[1,0] = 0
A_0_model[1,1] = -a_2_model

B_0_model = np.zeros((2,2))
B_0_model[0,0] = b_1_model
B_0_model[0,1] = c_1_model
B_0_model[1,0] = b_2_model
B_0_model[1,1] = c_2_model


A_0_actual = np.zeros((2,2))
A_0_actual[0,0] = -a_1_actual
A_0_actual[0,1] = 0
A_0_actual[1,0] = 0
A_0_actual[1,1] = -a_2_actual

B_0_actual = np.zeros((2,2))
B_0_actual[0,0] = b_1_actual
B_0_actual[0,1] = c_1_actual
B_0_actual[1,0] = b_2_actual
B_0_actual[1,1] = c_2_actual

# 离散化系数
A_d_model = expm(A_0_model * Ts)
expA_minus_I_model = expm(A_0_model * Ts) - np.eye(2)
B_d_model = np.linalg.solve(A_0_model, expA_minus_I_model @ B_0_model)

A_d_actual = expm(A_0_actual * Ts)
expA_minus_I_actual = expm(A_0_actual * Ts) - np.eye(2)
B_d_actual = np.linalg.solve(A_0_actual, expA_minus_I_actual @ B_0_actual)

# 状态矩阵
n = 2 * D + 2
A_model = np.zeros((n, n))
A_model[:2, :2] = A_d_model
A_model[0, n-2] = B_d_model[0, 0]  # 第 0 行的最后一列
A_model[1, n-2] = B_d_model[1, 0]  # 第 1 行的最后一列
A_model[0, n-1] = B_d_model[0, 1]
A_model[1, n-1] = B_d_model[1, 1]
for i in range(4, n):
    A_model[i, i - 2] = 1

A_actual = np.zeros((n, n))
A_actual[:2, :2] = A_d_actual
A_actual[0, n-2] = B_d_actual[0, 0]  # 第 0 行的最后一列
A_actual[1, n-2] = B_d_actual[1, 0]  # 第 1 行的最后一列
A_actual[0, n-1] = B_d_actual[0, 1]
A_actual[1, n-1] = B_d_actual[1, 1]
for i in range(4, n):
    A_actual[i, i - 2] = 1

b = np.zeros((n, 2))
b[2, 0] = 1
b[3, 1] = 1
c = np.zeros((2, n))
c[0, 0] = 1
c[1, 1] = 1

C = np.zeros((n,2))
C[0,0] = 1
C[1,1] = 1

# Fx 和 Gx
Fx = np.zeros((n * P, n))
Gx = np.zeros((2 * n * P, M))
for i in range(P):
    Fx[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A_model, i + 1)
    for j in range(min(i + 1, M)):
        Gx[ i * n * 2: (i + 1) * n * 2, j] = (np.linalg.matrix_power(A_model, i - j) @ b).flatten()
    if i >= M:
        Gx[ i * n * 2: (i + 1) * n * 2, M - 1] = sum((np.linalg.matrix_power(A_model, j) @ b).flatten() for j in range(i - M + 1))

# Fy 和 Gy
Fy = np.zeros((2 * P, n))
Gy = np.zeros((2 * P, 2 * M))
for i in range(P):
    Fy[2 * i: 2 * (i + 1), :] = c @ np.linalg.matrix_power(A_model, i + 1)
for i in range(P):
    for j in range(min(i + 1, M)):
        Gy[2 * i: 2 * (i + 1), 2 * j: 2 * (j + 1)] = c @ np.linalg.matrix_power(A_model, i - j) @ b
    if i >= M:
        Gy[2 * i: 2 * (i + 1), 2 * (M - 1): 2 * M] = sum(c @ np.linalg.matrix_power(A_model, j) @ b for j in range(i - M + 1))

# 初始化状态
x_model = np.zeros(n)
x_actual = np.zeros(n)


y_actual_history = np.zeros((2,total_steps))
u_history = np.zeros((2,total_steps))

for k in range(total_steps):
    if k == 0:
        u_previous = np.zeros((2, 1))
    else:
        u_previous = u_history[:, k - 1]
    model = gp.Model("qp")

    # 控制变量
    U_1 = model.addMVar(M, lb=u1_min, ub=u1_max, vtype=GRB.CONTINUOUS, name="U_1")
    U_2 = model.addMVar(M, lb=u2_min, ub=u2_max, vtype=GRB.CONTINUOUS, name="U_2")

    # 交替更新控制输入：将 U_1 和 U_2 交替放入控制变量
    U = model.addMVar(2*M, lb=min(u1_min,u2_min), ub=max(u1_max,u2_max), vtype=GRB.CONTINUOUS,name="U")

    # 交替赋值
    for i in range(M):
        model.addConstr(U[2 * i] == U_1[i])
        model.addConstr(U[2 * i + 1] == U_2[i])

    # numpy预计算：输出偏置
    y_offset = Fy @ x_model.T

    # Gurobi表达式：Gy@U是MLinExpr
    y_pred = y_offset + Gy @ U

    # 引入辅助变量 err
    err1 = model.addMVar(P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="err1")
    err2 = model.addMVar(P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="err2")
    for i in range(P):
        y_offset_1_i = float(y_offset[2 * i])
        gyu_1_i = Gy[2 * i, :] @ U
        model.addConstr(err1[i] == y_offset_1_i + gyu_1_i - r_1[k + i])

        y_offset_2_i = float(y_offset[2 * i + 1])
        gyu_2_i = Gy[2 * i + 1, :] @ U
        model.addConstr(err2[i] == y_offset_2_i + gyu_2_i - r_2[k + i])

    delta_u1 = model.addMVar(M, lb=delta_u1_min, ub=delta_u1_max, name="delta_u1")
    delta_u2 = model.addMVar(M, lb=delta_u2_min, ub=delta_u2_max, name="delta_u2")
    model.addConstr(delta_u1[0] == U[0] - u_previous[0])
    model.addConstr(delta_u2[0] == U[1] - u_previous[1])
    for i in range(M - 1):
         model.addConstr(delta_u1[i + 1] == U[2 * (i + 1)] - U[2 * i])
    for i in range(M - 1):
         model.addConstr(delta_u2[i + 1] == U[2 * i + 3] - U[2 * i + 1])

    # 引入二进制变量 p(i) 和 q(i)
    p1 = model.addMVar(M, vtype=GRB.BINARY, name="p1")
    q1 = model.addMVar(M, vtype=GRB.BINARY, name="q1")
    p2 = model.addMVar(M, vtype=GRB.BINARY, name="p2")
    q2 = model.addMVar(M, vtype=GRB.BINARY, name="q2")

    # 添加 p 和 q 的约束
    for i in range(M):
        model.addConstr(p1[i] + q1[i] <= 1)
        model.addConstr(delta_u1[i] <= negative_delta_u1_constrain * p1[i] + delta_u1_max * (1 - p1[i]))
        model.addConstr(delta_u1[i] >= positive_delta_u1_constrain * q1[i] + delta_u1_min * (1 - q1[i]))
        model.addConstr(delta_u1[i] <= delta_u1_max * q1[i])
        model.addConstr(delta_u1[i] >= delta_u1_min * p1[i])
        model.addConstr(p2[i] + q2[i] <= 1)
        model.addConstr(delta_u2[i] <= negative_delta_u2_constrain * p2[i] + delta_u2_max * (1 - p2[i]))
        model.addConstr(delta_u2[i] >= positive_delta_u2_constrain * q2[i] + delta_u2_min * (1 - q2[i]))
        model.addConstr(delta_u2[i] <= delta_u2_max * q2[i])
        model.addConstr(delta_u2[i] >= delta_u2_min * p2[i])


    # 构建代价函数
    cost = gp.QuadExpr()
    cost +=  err1 @ Q1 @ err1  # 加权的 err 项
    cost +=  err2 @ Q2 @ err2  # 加权的 err 项
    cost += delta_u1 @ R_delta1 @ delta_u1
    cost += delta_u2 @ R_delta2 @ delta_u2
    model.setObjective(cost, GRB.MINIMIZE)

    # 优化求解
    model.optimize()
    u_optimal = U.X

    print("u_optimal = ", u_optimal)

    # 应用第一个控制输入到实际系统
    unew = np.zeros((2,1))
    unew[0,0] = u_optimal[0]
    unew[1,0] = u_optimal[1]
    x_actual = x_actual @ A_actual.T + (b @ unew).T
    y_actual = x_actual @ C
    y_actual = y_actual.T

    # 添加干扰和噪声
    y_actual[0,0] += np.random.normal(0, external_disturbances)
    y_actual[1,0] += np.random.normal(0, external_disturbances)

    # 保存
    y_actual_history[:,k] = np.squeeze(y_actual)
    u_history[:,k] = np.squeeze(unew)

    # 更新模型状态
    x_model = x_actual

# 可视化
time = np.arange(total_steps)
r1_plot = r_1[:total_steps]  # 只取前 total_steps 个数据点
r2_plot = r_2[:total_steps]  # 只取前 total_steps 个数据点

plt.figure(figsize=(12, 12))

# 绘制 r_1 的图
plt.subplot(4, 1, 1)
plt.plot(time, y_actual_history[0, :], label='Actual Output (r_1)')
plt.plot(time, r1_plot, label='Reference (r_1)', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('System Output vs Reference for r_1')

# 绘制控制输入 u_1 对应 r_1 的图
plt.subplot(4, 1, 2)
plt.plot(time, u_history[0, :], label='Control Input (u_1)')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Input (u_1) for r_1')

# 绘制 r_2 的图
plt.subplot(4, 1, 3)
plt.plot(time, y_actual_history[1, :], label='Actual Output (r_2)')
plt.plot(time, r2_plot, label='Reference (r_2)', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('System Output vs Reference for r_2')

# 绘制控制输入 u_2 对应 r_2 的图
plt.subplot(4, 1, 4)
plt.plot(time, u_history[1, :], label='Control Input (u_2)')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Input (u_2) for r_2')

plt.tight_layout()
plt.show()
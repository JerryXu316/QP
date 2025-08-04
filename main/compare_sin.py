import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 定义系统参数
K_model = 0.8
T_model = 20
K_actual = 0.85
T_actual = 22
Ts = 1
D = 15
P = 50
M = 10
total_steps = 200
external_disturbances = 0.01

# 约束
u_min = -10.0
u_max = 10.0
delta_u_min = -5.0
delta_u_max = 5.0
positive_delta_u_constrain = 0.8
negative_delta_u_constrain = -0.8

# 定义代价函数的权重系数
Q = np.diag(np.ones(P) * 1.0)  # 对应 err 的权重矩阵
R = np.diag(np.ones(M) * 0.00)  # 对应 U 的权重矩阵
R_delta = np.diag(np.ones(M) * 0.001)  # 对应 U[i+1] - U[i] 的权重矩阵


# 控制目标
r = np.zeros(total_steps + P)
for i in range(total_steps + P):
    r[i] = 0.5 * np.sin(2 * np.pi * (1.0 / 100.0) * i) + 1.0

# 离散化系数
a_model = np.exp(-Ts / T_model)
b0_model = K_model * (1 - a_model)
a_actual = np.exp(-Ts / T_actual)
b0_actual = K_actual * (1 - a_actual)

# 状态矩阵
n = D + 1
A_model = np.zeros((n, n))
A_model[0, 0] = a_model
A_model[0, -1] = b0_model
for i in range(2, n):
    A_model[i, i - 1] = 1

A_actual = np.zeros((n, n))
A_actual[0, 0] = a_actual
A_actual[0, -1] = b0_actual
for i in range(2, n):
    A_actual[i, i - 1] = 1

b = np.zeros((n, 1))
b[1, 0] = 1
c = np.zeros((1, n))
c[0, 0] = 1

C = np.zeros(n)
C[0] = 1

# Fx 和 Gx
Fx = np.zeros((n * P, n))
Gx = np.zeros((n * P, M))
for i in range(P):
    Fx[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A_model, i + 1)
    for j in range(min(i + 1, M)):
        Gx[i * n:(i + 1) * n, j] = (np.linalg.matrix_power(A_model, i - j) @ b).flatten()
    if i >= M:
        Gx[i * n:(i + 1) * n, M - 1] = sum((np.linalg.matrix_power(A_model, j) @ b).flatten() for j in range(i - M + 1))

# Fy 和 Gy
Fy = np.zeros((P, n))
Gy = np.zeros((P, M))
for i in range(P):
    Fy[i, :] = c @ np.linalg.matrix_power(A_model, i + 1)
for i in range(P):
    for j in range(min(i + 1, M)):
        Gy[i, j] = (c @ np.linalg.matrix_power(A_model, i - j) @ b).item()
    if i >= M:
        Gy[i, M - 1] = sum((c @ np.linalg.matrix_power(A_model, j) @ b).item() for j in range(i - M + 1))

# 初始化状态
x_model = np.zeros(n)
x_actual = np.zeros(n)

y_actual_history_miqp = np.zeros(total_steps)
u_history_miqp = np.zeros(total_steps)

for k in range(total_steps):
    u_previous = u_history_miqp[k - 1] if k > 0 else 0
    # 创建模型
    model = gp.Model()

    # 控制变量
    U = model.addMVar(M, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="U")

    # 引入辅助变量 err
    err = model.addMVar(P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="err")

    # 引入二进制变量 p(i) 和 q(i)
    p = model.addMVar(M, vtype=GRB.BINARY, name="p")
    q = model.addMVar(M, vtype=GRB.BINARY, name="q")

    # 计算 y_offset
    y_offset = Fy @ x_model.T

    # 添加 err 的约束
    for i in range(P):
        model.addConstr(err[i] == y_offset[i] + Gy[i, :] @ U - r[k + i])

    # 添加 delta_u 的约束
    delta_u = model.addMVar(M, lb=delta_u_min, ub=delta_u_max, name="delta_u")
    model.addConstr(delta_u[0] == U[0] - u_previous)
    for i in range(1, M):
        model.addConstr(delta_u[i] == U[i] - U[i - 1])

    # 添加 p 和 q 的约束
    for i in range(M):
        model.addConstr(p[i] + q[i] <= 1)
        model.addConstr(delta_u[i] <= negative_delta_u_constrain * p[i] + delta_u_max * (1 - p[i]))
        model.addConstr(delta_u[i] >= positive_delta_u_constrain * q[i] + delta_u_min * (1 -q[i]))
        model.addConstr(delta_u[i] <= delta_u_max * q[i])
        model.addConstr(delta_u[i] >= delta_u_min * p[i])

    # 构建代价函数
    cost = gp.QuadExpr()
    cost += err @ Q @ err  # 加权的 err 项
    cost += U @ R @ U  # 加权的 U 项
    cost += delta_u @ R_delta @ delta_u
    model.setObjective(cost, GRB.MINIMIZE)

    # 优化求解
    model.optimize()
    u_optimal = U.X

    # 应用第一个控制输入到实际系统
    u = u_optimal[0]
    x_actual = x_actual @ A_actual.T + (b * u).T
    y_actual = x_actual @ C
    y_actual = float(np.squeeze(y_actual))

    # 添加干扰和噪声
    y_actual += np.random.normal(0, external_disturbances)

    # 保存
    y_actual_history_miqp[k] = y_actual
    u_history_miqp[k] = u

    # 更新模型状态
    x_model = x_actual

# 初始化状态
x_model = np.zeros(n)
x_actual = np.zeros(n)


y_actual_history = np.zeros(total_steps)
u_history = np.zeros(total_steps)

for k in range(total_steps):
    u_previous = u_history[k - 1]
    model = gp.Model("qp")

    # 控制变量
    U = model.addMVar(M, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="U")

    # numpy预计算：输出偏置
    y_offset = Fy @ x_model.T  # numpy array，长度P

    # Gurobi表达式：Gy@U是MLinExpr
    y_pred = y_offset + Gy @ U  # 预测输出，长度P

    # 引入辅助变量 err
    err = model.addMVar(P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="err")
    for i in range(P):
        y_offset_i = float(y_offset[i])
        gyu_i = Gy[i, :] @ U
        model.addConstr(err[i] == y_offset_i + gyu_i - r[k + i])

    delta_u = model.addMVar(M, lb=delta_u_min, ub=delta_u_max, name="delta_u")
    model.addConstr(delta_u[0] == U[0] - u_previous)
    for i in range(M - 1):
         model.addConstr(delta_u[i + 1] == U[i + 1] - U[i])

    # 构建代价函数
    cost = gp.QuadExpr()
    cost +=  err @ Q @ err  # 加权的 err 项
    cost += U @ R @ U  # 加权的 U 项
    cost += delta_u @ R_delta @ delta_u
    model.setObjective(cost, GRB.MINIMIZE)

    # 优化求解
    model.optimize()
    u_optimal = U.X

    # 应用第一个控制输入到实际系统
    u = u_optimal[0]
    x_actual = x_actual @ A_actual.T + (b * u).T
    y_actual = x_actual @ C
    y_actual = float(np.squeeze(y_actual))

    # 添加干扰和噪声
    y_actual += np.random.normal(0, external_disturbances)

    # 保存
    y_actual_history[k] = y_actual
    u_history[k] = u

    # 更新模型状态
    x_model = x_actual

# 可视化
time = np.arange(total_steps)
r_plot = r[:total_steps]  # 只取前 total_steps 个数据点

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, y_actual_history_miqp, color='tab:orange', label='Discrete MPC')
plt.plot(time, y_actual_history,      color='tab:blue',  label='Continuous MPC')
plt.plot(time, r_plot, 'g--', label='Reference')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('System Output vs Reference')

plt.subplot(2, 1, 2)
plt.step(time, u_history_miqp, color='tab:orange', label='Discrete MPC')
plt.step(time, u_history, color='tab:blue',  label='Continuous MPC')
plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Input')

plt.tight_layout()
plt.show()
import numpy as np
import control as ctrl
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# 定义系统参数
Ts = 1
D = 15
P = 50
M = 10
total_steps = 100
external_disturbances = 0.0

# 约束
u_min = -3.0
u_max = 3.0
delta_u_min = -1.5
delta_u_max = 1.5
positive_delta_u_constrain = 0.4
negative_delta_u_constrain = -0.4


# 定义代价函数的权重系数
Q = np.diag(np.ones(P) * 1.0)  # 对应 err 的权重矩阵
R = np.diag(np.ones(M) * 0.0)  # 对应 U 的权重矩阵
R_delta = np.diag(np.ones(M) * 0.001)  # 对应 U[i+1] - U[i] 的权重矩阵


# 控制目标
r = np.ones(total_steps + P) * 1.0

# ------------------ 系统参数 ------------------
b_1_model, b_0_model = 2.0, 1.0   # 分子系数 [b_m, b_{m-1}, ..., b_0]
a_2_model, a_1_model, a_0_model = 1.0, 3.0, 2.0   # 分母系数 [a_n, a_{n-1}, ..., a_0]

# 连续时间传递函数
num_model = [b_1_model, b_0_model]
den_model = [a_2_model, a_1_model, a_0_model]
sys_tf_model = ctrl.TransferFunction(num_model, den_model)

# 转换为连续时间状态空间
sys_ss_model = ctrl.ss(sys_tf_model)


sys_d_model = ctrl.c2d(sys_ss_model, Ts, method='zoh')   # 零阶保持法离散化（也可以选择其他的方法）

# 提取离散状态空间矩阵
A_d_model, B_d_model, C_d_model, D_d_model = sys_d_model.A, sys_d_model.B, sys_d_model.C, sys_d_model.D
print("离散状态空间矩阵：")
print("Ad_model =\n", A_d_model, "\nBd_model =\n", B_d_model, "\nCd_model =", C_d_model, "\nDd_model =", D_d_model)

# ------------------ 实际参数 ------------------
b_1_actual, b_0_actual = 2.0, 1.0   # 分子系数 [b_m, b_{m-1}, ..., b_0]
a_2_actual, a_1_actual, a_0_actual = 1.0, 3.0, 2.0   # 分母系数 [a_n, a_{n-1}, ..., a_0]

# 连续时间传递函数
num_actual = [b_1_actual, b_0_actual]
den_actual = [a_2_actual, a_1_actual, a_0_actual]
sys_tf_actual = ctrl.TransferFunction(num_actual, den_actual)

# 转换为连续时间状态空间
sys_ss_actual = ctrl.ss(sys_tf_actual)


sys_d_actual = ctrl.c2d(sys_ss_actual, Ts, method='zoh')   # 零阶保持法离散化（也可以选择其他的方法）

# 提取离散状态空间矩阵
A_d_actual, B_d_actual, C_d_actual, D_d_actual = sys_d_actual.A, sys_d_actual.B, sys_d_actual.C, sys_d_actual.D
print("离散状态空间矩阵：")
print("Ad_actual =\n", A_d_actual, "\nBd_actual =\n", B_d_actual, "\nCd_actual =", C_d_actual, "\nDd_actual =", D_d_actual)


# 状态矩阵
n = D + 2
A_model = np.zeros((n, n))
A_model[:2, :2] = A_d_model
A_model[0, n-1] = B_d_model[0, 0]  # 第 0 行的最后一列
A_model[1, n-1] = B_d_model[1, 0]  # 第 1 行的最后一列
for i in range(3, n):
    A_model[i, i - 1] = 1

A_actual = np.zeros((n, n))
A_actual[:2, :2] = A_d_actual
A_actual[0, n-1] = B_d_actual[0, 0]  # 第 0 行的最后一列
A_actual[1, n-1] = B_d_actual[1, 0]  # 第 1 行的最后一列
for i in range(3, n):
    A_actual[i, i - 1] = 1

b = np.zeros((n, 1))
b[2, 0] = 1

C_model = np.zeros((1, n))
C_model[0, 0] = C_d_model[0, 0]
C_model[0, 1] = C_d_model[0, 1]

C_actual = np.zeros((1, n))
C_actual[0, 0] = C_d_actual[0, 0]
C_actual[0, 1] = C_d_actual[0, 1]



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
    Fy[i, :] = C_model @ np.linalg.matrix_power(A_model, i + 1)
for i in range(P):
    for j in range(min(i + 1, M)):
        Gy[i, j] = (C_model @ np.linalg.matrix_power(A_model, i - j) @ b).item()
    if i >= M:
        Gy[i, M - 1] = sum((C_model @ np.linalg.matrix_power(A_model, j) @ b).item() for j in range(i - M + 1))

# 初始化状态
x_model = np.zeros(n)
x_actual = np.zeros(n)


y_actual_history_miqp = np.zeros(total_steps)
u_history_miqp = np.zeros(total_steps)
flag = np.zeros(total_steps)

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
        model.addConstr(delta_u[i] >= positive_delta_u_constrain * q[i] + delta_u_min * (1 - q[i]))
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
    y_actual = x_actual @ C_actual.T
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
flag = np.zeros(total_steps)

y_actual_history = np.zeros(total_steps)
u_history = np.zeros(total_steps)
u_actual = np.zeros(total_steps)

for k in range(total_steps):
    u_previous = u_history[k - 1] if k > 0 else 0
    u_previous_actual = u_actual[k - 1] if k > 0 else 0
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
    du = u_optimal[0] - u_previous_actual
    if du == 0.0:
        u = u_previous_actual
        flag[k] = 0.0
    elif negative_delta_u_constrain < du < positive_delta_u_constrain:
        u = u_previous_actual
        flag[k] = 1.0
    else:
        u = u_optimal[0]
        flag[k] = 0.0



    x_model = x_model @ A_actual.T + (b * u_optimal[0]).T
    x_actual = x_actual @ A_actual.T + (b * u).T

    y_actual = x_actual @ C_actual.T
    y_actual = float(np.squeeze(y_actual))

    # 添加干扰和噪声
    y_actual += np.random.normal(0, external_disturbances)

    # 保存
    y_actual_history[k] = y_actual
    u_actual[k] = u
    u_history[k] = u_optimal[0]


# 可视化
time = np.arange(total_steps)


plt.figure(figsize=(12, 6))

# 输出
plt.subplot(2, 1, 1)
plt.plot(time, y_actual_history_miqp, color='tab:orange', label='Discrete MPC')
plt.plot(time, y_actual_history, color='tab:blue', label='Continuous MPC')
plt.plot(time, r[:total_steps], 'g--', label='Reference')

# # 红点：flag=1 处的 Continuous MPC
# red_idx = np.where(flag == 1)[0]
# if red_idx.size:
#     plt.scatter(red_idx, y_actual_history[red_idx], color='red', s=20, zorder=5)

plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('System Output vs Reference')

# 控制量
plt.subplot(2, 1, 2)
# plt.step(time, u_history_miqp, color='tab:orange', label='Discrete MPC')
plt.step(time, u_history, color='tab:blue', label='Continuous MPC(model)')
plt.step(time,u_actual, color='tab:red', label='Continuous MPC(actual)')

# # 红点：flag=1 处的 Continuous MPC 控制量
# if red_idx.size:
#     plt.scatter(red_idx, u_history[red_idx], color='red', s=20, zorder=5)

plt.xlabel('Time Step')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Input')

plt.tight_layout()
plt.show()
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# 系统参数
Ts = 1
tau = 20
K = 0.8
a = np.exp(-Ts / tau)
b = K * tau * (1 - a)
# MPC参数
Np = 50
Nc = 10
delay = 15
N_sim = 500
ref = 50.0

# Big-M参数（要比控制范围大）
M = 1000

# 初始化
y = np.zeros(N_sim + 1)
u = np.zeros(N_sim + Np + delay + 1)

# 权重矩阵
Q = np.eye(Np)
R = 0.01 * np.eye(Nc)

for t in range(N_sim):
    # MPC控制变量
    u_control = cp.Variable(Nc)
    delta_u = cp.Variable(Nc)
    z_pos = cp.Variable(Nc, boolean=True)  # Δu >= 0.8
    z_neg = cp.Variable(Nc, boolean=True)  # Δu <= -0.8

    # 预测y序列
    y_pred = []
    prev_y = y[t]

    # 构建完整控制序列u_full（历史+预测）
    u_full = []
    for i in range(Np):
        idx = t + i - delay
        if idx < 0:
            u_full.append(0.0)
        elif idx < t:
            u_full.append(u[idx])
        elif idx - t < Nc:
            u_full.append(u_control[idx - t])
        else:
            u_full.append(0.0)

    # 用a,b递推y_pred
    for i in range(Np):
        y_next = a * prev_y + b * u_full[i]
        y_pred.append(y_next)
        prev_y = y_next

    y_pred_expr = cp.hstack(y_pred)
    e = y_pred_expr - ref

    # 计算delta_u = u_control - u[t-1]
    # 注意第一个delta_u对应u_control[0] - u[t-1]
    # 后面delta_u[i] = u_control[i] - u_control[i-1]
    delta_u_expr = []
    for i in range(Nc):
        if i == 0:
            if t - 1 >= 0:
                delta_u_expr.append(u_control[0] - u[t-1])
            else:
                delta_u_expr.append(u_control[0])  # t=0时，前值视为0
        else:
            delta_u_expr.append(u_control[i] - u_control[i-1])
    delta_u_expr = cp.hstack(delta_u_expr)

    # 约束列表
    constraints = []

    # delta_u定义等式
    for i in range(Nc):
        constraints.append(delta_u[i] == delta_u_expr[i])

    # delta_u取值约束：只允许 delta_u=0 或 delta_u >=0.8 或 delta_u <= -0.8
    for i in range(Nc):
        # 互斥：z_pos[i] + z_neg[i] <= 1
        constraints.append(z_pos[i] + z_neg[i] <= 1)

        # delta_u >= 0.8 * z_pos[i] - M * (1 - z_pos[i])
        constraints.append(delta_u[i] >= 0.8 * z_pos[i] - M * (1 - z_pos[i]))
        # delta_u <= M * z_pos[i]
        constraints.append(delta_u[i] <= M * z_pos[i])

        # delta_u <= -0.8 * z_neg[i] + M * (1 - z_neg[i])
        constraints.append(delta_u[i] <= -0.8 * z_neg[i] + M * (1 - z_neg[i]))
        # delta_u >= -M * z_neg[i]
        constraints.append(delta_u[i] >= -M * z_neg[i])

    # 代价函数
    cost = cp.quad_form(e, Q) + cp.quad_form(u_control, R)

    # 建立问题
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # 求解并计时
    start_time = time.time()
    prob.solve(solver=cp.ECOS_BB, verbose=False)
    solve_time = time.time() - start_time

    # 取控制输入第一个值作为实际控制
    if u_control.value is not None:
        u[t] = u_control.value[0]
        delta_u_val = delta_u.value
    else:
        u[t] = 0
        delta_u_val = np.zeros(Nc)

    # 系统状态更新
    u_delay_val = u[t - delay] if t - delay >= 0 else 0
    y[t+1] = a * y[t] + b * u_delay_val

    # 打印
    print(f"------------------------------------------------------------")
    print(f"t={t}, solve_time={solve_time:.4f}s")
    print(f"  Δu = {np.round(delta_u_val, 4)}")
    print(f"  u(t) = {u[t]:.4f}")
    print(f"  y(t+1) = {y[t+1]:.4f}")

# 绘图
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(range(N_sim + 1), y, label="Output y", linewidth=2)
plt.axhline(ref, color='gray', linestyle='--', label="Reference")
plt.ylabel("y(t)")
plt.title("系统输出")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.step(range(N_sim), u[:N_sim], where='post', label="Control u", linewidth=2, color='orange')
plt.ylabel("u(t)")
plt.xlabel("时间步")
plt.title("控制输入")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

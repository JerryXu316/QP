import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 系统参数
Ts = 1
tau = 20
K = 0.8
a = np.exp(-Ts / tau)
b = K * (1 - a)

# MPC参数
Np = 50          # 预测步数
delay = 15       # 延迟步数
Nc = 10           # 控制步数
N_sim = 100      # 仿真步数
ref = 1     # 目标值

# 初始化
y = np.zeros(N_sim + 1)
u = np.zeros(N_sim + Np + delay + 1)

# Q、R矩阵
Q = np.eye(Np)
R = 0.01 * np.eye(Nc + 1)

for t in range(N_sim):
    y_pred = []
    prev_y = y[t]
    u_control = cp.Variable(Nc + 1)

    # 预测Np步
    for i in range(1, Np + 1):
        u_index = t + i - delay - 1  # 注意i从1开始

        if u_index < 0:
            u_eff = 0  # 延迟超前，取0
        elif u_index < t:
            u_eff = u[u_index]  # 历史控制值
        elif (i-delay-1) < Nc :  # 预测第i步用u_control(i-delay)，即第i-delay项
            u_eff = u_control[i-delay-1]  # 注意i从1开始，所以i-delay-1是实际的控制步数
        else:
            u_eff = 0  # 超出控制域部分默认0

        y_next = a * prev_y + b * u_eff
        y_pred.append(y_next)
        prev_y = y_next

    # 误差向量
    y_pred_expr = cp.hstack(y_pred)
    e = y_pred_expr - ref

    # 构造代价函数
    cost = cp.quad_form(e, Q) + cp.quad_form(u_control, R)

    # QP求解
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver=cp.OSQP)

    # 更新控制量
    if u_control.value is not None:
        u[t] = u_control.value[0]  # 取第一步控制增量
    else:
        u[t] = 0
    # 输出u_control的全部值
    if u_control.value is not None:
        print(f"Time step {t}: u_control = {u_control.value}")
        u[t] = u_control.value[0]  # 取第一步控制增量
    else:
        print(f"Time step {t}: u_control is None")
        u[t] = 0
    # 系统差分方程更新
    u_delay = u[t - delay] if t - delay >= 0 else 0
    y[t+1] = a * y[t] + b * u_delay

# 绘图
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(y[:N_sim], label="y(t)", linewidth=2)
plt.axhline(ref, linestyle='--', color='gray', label="reference")
plt.ylabel("Output y")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.step(range(N_sim), u[:N_sim], label="u(t)", where='post')
plt.ylabel("Control u")
plt.xlabel("Time step")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
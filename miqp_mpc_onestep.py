import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# =================  系统参数  =================
Ts   = 1.0
tau  = 20.0
K    = 0.8
a    = np.exp(-Ts / tau)
b    = K * (1 - a)

# =================  MPC 参数  =================
Np      = 50          # 预测步长
delay   = 15          # 纯延迟
N_sim   = 500         # 仿真步数

# 三段 Δu 离散约束
Δu_min  = 0.8
M       = 1000        # Big-M

# =================  参考信号  =================
A     = 5.0
omega = 2 * np.pi / 300
C     = 10.0

# =================  初始化  =================
y = np.zeros(N_sim + 1)
u = np.zeros(N_sim + Np + delay + 1)   # 预留历史及未来

# 权重
Q = np.eye(Np)
R = 0.01

# =================  仿真主循环  =================
for t in range(N_sim):
    # 决策变量：仅一个标量 u_fix
    u_fix   = cp.Variable()
    delta_u = cp.Variable()
    z_pos   = cp.Variable(boolean=True)
    z_neg   = cp.Variable(boolean=True)

    # 构造参考序列
    ref_t = np.array([A * np.sin(omega * (t + i)) + C
                      for i in range(1, Np + 1)])

    # 构造恒定输入序列 u_pred_vec
    u_pred_vec = []
    for i in range(Np):
        idx = t + i - delay
        if idx < 0:
            u_pred_vec.append(0.0)
        elif idx < t:
            u_pred_vec.append(u[idx])
        else:                     # 预测区间恒等于 u_fix
            u_pred_vec.append(u_fix)
    u_pred_vec = np.array(u_pred_vec)

    # 一步递推得到 y_pred
    y_pred = []
    prev_y = y[t]
    for u_k in u_pred_vec:
        prev_y = a * prev_y + b * u_k
        y_pred.append(prev_y)
    y_pred_expr = cp.hstack(y_pred)
    e = y_pred_expr - ref_t

    # 计算 Δu（u_fix 与上一时刻实际输入之差）
    u_prev = u[t-1] if t > 0 else 0.0
    delta_u_expr = u_fix - u_prev

    # 约束
    constraints = [
        delta_u == delta_u_expr,
        z_pos + z_neg <= 1,
        delta_u >=  Δu_min * z_pos - M * (1 - z_pos),
        delta_u <=  M * z_pos,
        delta_u <= -Δu_min * z_neg + M * (1 - z_neg),
        delta_u >= -M * z_neg
    ]

    # 目标函数
    cost = cp.quad_form(e, Q) + R * u_fix**2   # 单变量可直接平方
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # 求解
    start = time.time()
    prob.solve(solver=cp.ECOS_BB, verbose=False)
    solve_time = time.time() - start

    # 更新控制量
    if u_fix.value is not None:
        u[t] = float(u_fix.value)
    else:
        u[t] = 0.0

    # 系统更新（考虑纯延迟）
    u_delay = u[t - delay] if t - delay >= 0 else 0.0
    y[t + 1] = a * y[t] + b * u_delay

    # 实时打印
    print(f"t={t:3d} | Δu={delta_u.value:7.3f} | u={u[t]:7.3f} "
          f"| y={y[t+1]:7.3f} | solve={solve_time:.4f}s")

# =================  绘图  =================
plt.figure(figsize=(10, 6))
t_axis = np.arange(N_sim + 1)

plt.subplot(2, 1, 1)
plt.plot(t_axis, y, label='y(t)', linewidth=2)
ref_plot = A * np.sin(omega * t_axis) + C
plt.plot(t_axis, ref_plot, '--', color='gray', label='reference')
plt.ylabel('Output y')
plt.legend(); plt.grid()

plt.subplot(2, 1, 2)
plt.step(t_axis[:-1], u[:N_sim], where='post', label='u(t)')
plt.ylabel('Control u'); plt.xlabel('Time step')
plt.legend(); plt.grid()
plt.tight_layout()
plt.show()
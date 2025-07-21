import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import torch

# 加载训练好的模型
class MIQPNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MIQPNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 加载模型
model = MIQPNet(input_dim=20, output_dim=20)  # 假设输入维度为20（10个y和10个u），输出维度为20（10个z_pos和10个z_neg）
model.load_state_dict(torch.load("miqp_model.pt"))
model.eval()

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

# 正弦参考信号参数
A = 5          # 振幅
omega = 2 * np.pi / 120  # 角频率，周期 T = 60
C = 10           # 偏置
T = 120  # 周期

# 初始化
y_miqp = np.random.uniform(0, 20, N_sim + 1)  # y的初值在0-20之间
u_miqp = np.zeros(N_sim + Np + delay + 1)
y_nnp = np.copy(y_miqp)
u_nnp = np.copy(u_miqp)
phi = np.random.uniform(-T/2, T/2)  # 相位偏移量在-T/2到T/2之间


# 仿真过程
solve_times_miqp = []
solve_times_nnp = []

for t in range(N_sim):
    # === MIQP 控制过程 ===
    start_time = time.time()
    u_control_miqp = cp.Variable(Nc)
    delta_u_miqp = cp.Variable(Nc)
    z_pos_miqp = cp.Variable(Nc, boolean=True)
    z_neg_miqp = cp.Variable(Nc, boolean=True)

    y_pred_miqp = []
    prev_y = y_miqp[t]
    ref_t = np.array([A * np.sin(omega * (t + i)) + C for i in range(1, Np + 1)])

    u_full_miqp = []
    for i in range(Np):
        idx = t + i - delay
        if idx < 0:
            u_full_miqp.append(0.0)
        elif idx < t:
            u_full_miqp.append(u_miqp[idx])
        elif idx - t < Nc:
            u_full_miqp.append(u_control_miqp[idx - t])
        else:
            u_full_miqp.append(0.0)

    for i in range(Np):
        y_next = a * prev_y + b * u_full_miqp[i]
        y_pred_miqp.append(y_next)
        prev_y = y_next

    y_pred_expr_miqp = cp.hstack(y_pred_miqp)
    e_miqp = y_pred_expr_miqp - ref_t

    delta_u_expr_miqp = []
    for i in range(Nc):
        if i == 0:
            delta_u_expr_miqp.append(u_control_miqp[0] - (u_miqp[t-1] if t-1 >= 0 else 0))
        else:
            delta_u_expr_miqp.append(u_control_miqp[i] - u_control_miqp[i-1])
    delta_u_expr_miqp = cp.hstack(delta_u_expr_miqp)

    constraints_miqp = []
    for i in range(Nc):
        constraints_miqp.append(delta_u_miqp[i] == delta_u_expr_miqp[i])
        constraints_miqp.append(z_pos_miqp[i] + z_neg_miqp[i] <= 1)
        constraints_miqp.append(delta_u_miqp[i] >= 0.8 * z_pos_miqp[i] - 1000 * (1 - z_pos_miqp[i]))
        constraints_miqp.append(delta_u_miqp[i] <= 1000 * z_pos_miqp[i])
        constraints_miqp.append(delta_u_miqp[i] <= -0.8 * z_neg_miqp[i] + 1000 * (1 - z_neg_miqp[i]))
        constraints_miqp.append(delta_u_miqp[i] >= -1000 * z_neg_miqp[i])

    cost_miqp = cp.quad_form(e_miqp, np.eye(Np)) + cp.quad_form(u_control_miqp, 0.01 * np.eye(Nc))
    prob_miqp = cp.Problem(cp.Minimize(cost_miqp), constraints_miqp)

    prob_miqp.solve(solver=cp.ECOS_BB, verbose=False)
    solve_time_miqp = time.time() - start_time
    solve_times_miqp.append(solve_time_miqp)

    if u_control_miqp.value is not None:
        u_miqp[t] = u_control_miqp.value[0]
    u_delay_val_miqp = u_miqp[t - delay] if t - delay >= 0 else 0
    y_miqp[t+1] = a * y_miqp[t] + b * u_delay_val_miqp

    # === 基于神经网络的 QP 控制过程（阈值 0.5 版本）===
    start_time = time.time()

    # 1. 构造输入
    y_seq = y_nnp[max(0, t - 10):t]
    u_seq = u_nnp[max(0, t - 10):t]
    y_seq = np.pad(y_seq, (10 - len(y_seq), 0), mode='constant')
    u_seq = np.pad(u_seq, (10 - len(u_seq), 0), mode='constant')
    x = np.concatenate([y_seq, u_seq])
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # 2. 神经网络推理 + 阈值化
    with torch.no_grad():
        logits = model(x).numpy().flatten()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    z_pos_bin = (probs[:Nc] >= 0.5).astype(int)
    z_neg_bin = (probs[Nc:] >= 0.5).astype(int)

    # 3. 建立 QP 变量
    u_control_nnp = cp.Variable(Nc)
    delta_u_nnp = cp.Variable(Nc)

    # 4. 预测未来 y（与原来一致）
    prev_y = y_nnp[t]
    ref_t = np.array([A * np.sin(omega * (t + k + 1)) + C for k in range(Np)])
    u_full = []
    for k in range(Np):
        idx = t + k - delay
        if idx < 0:
            u_full.append(0.0)
        elif idx < t:
            u_full.append(u_nnp[idx])
        elif idx - t < Nc:
            u_full.append(u_control_nnp[idx - t])
        else:
            u_full.append(0.0)

    y_pred = []
    for k in range(Np):
        prev_y = a * prev_y + b * u_full[k]
        y_pred.append(prev_y)
    y_pred_expr = cp.hstack(y_pred)
    e = y_pred_expr - ref_t

    # 5. 目标函数
    cost = cp.quad_form(e, np.eye(Np)) + cp.quad_form(u_control_nnp, 0.01 * np.eye(Nc))

    # 6. 约束：阈值化后的硬约束 + Δu 定义
    constraints = []
    for i in range(Nc):
        if z_pos_bin[i] == 1 and z_neg_bin[i] == 0:
            constraints.append(delta_u_nnp[i] >= 0.8)
        elif z_pos_bin[i] == 0 and z_neg_bin[i] == 1:
            constraints.append(delta_u_nnp[i] <= -0.8)
        else:  # 其余情况不额外限制
            pass

    for i in range(Nc):
        if i == 0:
            prev_u = u_nnp[t - 1] if t - 1 >= 0 else 0.0
        else:
            prev_u = u_control_nnp[i - 1]
        constraints.append(delta_u_nnp[i] == u_control_nnp[i] - prev_u)

    # 7. 求解
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    solve_time_nnp = time.time() - start_time
    solve_times_nnp.append(solve_time_nnp)

    # 8. 应用控制量
    if u_control_nnp.value is not None:
        u_nnp[t] = u_control_nnp.value[0]
    u_delay_val = u_nnp[t - delay] if t - delay >= 0 else 0.0
    y_nnp[t + 1] = a * y_nnp[t] + b * u_delay_val

    print(f"t={t}, MIQP time={solve_time_miqp:.3f}s, QP time={solve_time_nnp:.3f}s")

# 可视化
plt.figure(figsize=(12, 10))

# 绘制系统输出
plt.subplot(3, 1, 1)
plt.plot(range(N_sim + 1), y_miqp, label="MIQP Output y", linewidth=2)
plt.plot(range(N_sim + 1), y_nnp, label="NN+QP Output y", linewidth=2, linestyle='--')
t = np.arange(N_sim + 1)
ref_plot = A * np.sin(omega * t ) + C
plt.plot(t, ref_plot, '--', color='gray', label="Reference")
plt.ylabel("y(t)")
plt.title("System output")
plt.legend()
plt.grid()

# 绘制控制输入
plt.subplot(3, 1, 2)
plt.step(range(N_sim), u_miqp[:N_sim], label="MIQP Control u", where='post')
plt.step(range(N_sim), u_nnp[:N_sim], label="NN+QP Control u", where='post', linestyle='--')
plt.ylabel("u(t)")
plt.xlabel("time step")
plt.title("Control input")
plt.legend()
plt.grid()

# 绘制求解时间
plt.subplot(3, 1, 3)
plt.plot(range(N_sim), solve_times_miqp, label="MIQP Solve Time", linewidth=2)
plt.plot(range(N_sim), solve_times_nnp, label="NN+QP Solve Time", linewidth=2, linestyle='--')
plt.ylabel("Solve Time (s)")
plt.xlabel("time step")
plt.title("solution time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
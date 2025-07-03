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
b = K * (1 - a)

# MPC参数
Np = 50
Nc = 10
delay = 15
N_sim = 200
ref = 50.0

# 正弦参考信号参数
A = 5
omega = 2 * np.pi / 60
T = 60  # 周期

# 初始化
y = np.random.uniform(0, 20, N_sim + 1)  # y的初值在0-20之间
u = np.zeros(N_sim + Np + delay + 1)
phi = np.random.uniform(-T/2, T/2)  # 相位偏移量在-T/2到T/2之间
C = 10  # 偏移量C保持不变

# 仿真过程
solve_times = []
y_pred_list = []
u_pred_list = []

for t in range(N_sim):
    # 准备输入数据
    y_seq = y[max(0, t - 10):t]
    u_seq = u[max(0, t - 10):t]

    # 填充零以确保长度为10
    y_seq = np.pad(y_seq, (10 - len(y_seq), 0), mode='constant')
    u_seq = np.pad(u_seq, (10 - len(u_seq), 0), mode='constant')

    x = np.concatenate([y_seq, u_seq])
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # 使用模型预测 z_pos 和 z_neg
    with torch.no_grad():
        predictions = model(x).numpy().flatten()
    z_pos_pred = predictions[:Nc]
    z_neg_pred = predictions[Nc:]

    # 构建 QP 问题
    u_control = cp.Variable(Nc)
    delta_u = cp.Variable(Nc)
    y_pred = []
    prev_y = y[t]
    ref_t = np.array([A * np.sin(omega * (t + i)) + C for i in range(1, Np + 1)])

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

    for i in range(Np):
        y_next = a * prev_y + b * u_full[i]
        y_pred.append(y_next)
        prev_y = y_next

    y_pred_expr = cp.hstack(y_pred)
    e = y_pred_expr - ref_t

    # 代价函数
    cost = cp.quad_form(e, np.eye(Np)) + cp.quad_form(u_control, 0.01 * np.eye(Nc))

    # 约束条件
    constraints = []
    for i in range(Nc):
        if z_pos_pred[i] == 1 and z_neg_pred[i] == 0:
            constraints.append(delta_u[i] >= 0.8)
        elif z_pos_pred[i] == 0 and z_neg_pred[i] == 1:
            constraints.append(delta_u[i] <= -0.8)
        elif z_pos_pred[i] == 0 and z_neg_pred[i] == 0:
            constraints.append(delta_u[i] == 0)

    # 计算 delta_u
    for i in range(Nc):
        if i == 0:
            constraints.append(delta_u[i] == u_control[i] - (u[t-1] if t-1 >= 0 else 0))
        else:
            constraints.append(delta_u[i] == u_control[i] - u_control[i-1])

    # 求解 QP 问题
    prob = cp.Problem(cp.Minimize(cost), constraints)
    start_time = time.time()
    prob.solve(solver=cp.OSQP, verbose=False)
    solve_time = time.time() - start_time

    if u_control.value is not None:
        u[t] = u_control.value[0]
    else:
        u[t] = 0

    u_delay_val = u[t - delay] if t - delay >= 0 else 0
    y[t+1] = a * y[t] + b * u_delay_val

    solve_times.append(solve_time)
    y_pred_list.append(y_pred_expr.value)
    u_pred_list.append(u_control.value)

    print(f"t={t}, time={solve_time:.3f}s")

# 可视化
plt.figure(figsize=(12, 10))

# 绘制系统输出
plt.subplot(3, 1, 1)
plt.plot(range(N_sim + 1), y, label="Output y", linewidth=2)
t = np.arange(N_sim + 1)
ref_plot = A * np.sin(omega * t ) + C
plt.plot(t, ref_plot, '--', color='gray', label="Reference")
plt.ylabel("y(t)")
plt.title("系统输出")
plt.legend()
plt.grid()

# 绘制控制输入
plt.subplot(3, 1, 2)
plt.step(range(N_sim), u[:N_sim], label="Control u", where='post')
plt.ylabel("u(t)")
plt.xlabel("时间步")
plt.title("控制输入")
plt.legend()
plt.grid()

# 绘制求解时间
plt.subplot(3, 1, 3)
plt.plot(range(N_sim), solve_times, label="Solve Time", linewidth=2)
plt.ylabel("Solve Time (s)")
plt.xlabel("时间步")
plt.title("求解时间")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
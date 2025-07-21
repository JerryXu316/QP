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
b = K * tau * (1 - a)
# ========= MPC =========
Np, Nc, delay = 50, 10, 15
N_sim = 500
Δu_min, M = 0.8, 1000.0
Q, R = np.eye(Np), 0.01

# ========= smooth reference =========
A1, A2 = 5.0, 3.0
T1, T2 = 300.0, 120.0
k_quad = 0.00005          # 二次项斜率
C_smooth = 10.0

y = np.zeros(N_sim + 1)
u = np.zeros(N_sim + Np + delay + 1)

for t in range(N_sim):
    u_fix   = cp.Variable()
    delta_u = cp.Variable()
    z_pos   = cp.Variable(boolean=True)
    z_neg   = cp.Variable(boolean=True)

    # 光滑参考：双频正弦 + 缓坡二次
    ref_t = np.array([
        A1 * np.sin(2 * np.pi * (t + i) / T1) +
        A2 * np.sin(2 * np.pi * (t + i) / T2) +
        k_quad * (t + i)**2 +
        C_smooth
        for i in range(1, Np + 1)
    ])

    # MPC 预测不变
    u_pred_vec = []
    for i in range(Np):
        idx = t + i - delay
        u_pred_vec.append(u[idx] if 0 <= idx < t else (u_fix if idx >= t else 0.0))
    y_pred = []
    prev_y = y[t]
    for val in u_pred_vec:
        prev_y = a * prev_y + b * val
        y_pred.append(prev_y)
    e = cp.hstack(y_pred) - ref_t

    u_prev = u[t - 1] if t > 0 else 0.0
    constraints = [
        delta_u == u_fix - u_prev,
        z_pos + z_neg <= 1,
        delta_u >=  Δu_min * z_pos - M * (1 - z_pos),
        delta_u <=  M * z_pos,
        delta_u <= -Δu_min * z_neg + M * (1 - z_neg),
        delta_u >= -M * z_neg
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(e, Q) + R * u_fix**2), constraints)
    prob.solve(solver=cp.ECOS_BB, verbose=False)

    u[t] = float(u_fix.value) if u_fix.value is not None else 0.0
    y[t + 1] = a * y[t] + b * (u[t - delay] if t - delay >= 0 else 0.0)

# ========= plot =========
t_axis = np.arange(N_sim + 1)
ref_plot = (
    A1 * np.sin(2 * np.pi * t_axis / T1) +
    A2 * np.sin(2 * np.pi * t_axis / T2) +
    k_quad * t_axis**2 +
    C_smooth
)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_axis, y, label='y(t)', linewidth=2)
plt.plot(t_axis, ref_plot, '--', color='gray', label='reference')
plt.ylabel('Output'); plt.legend(); plt.grid()

plt.subplot(2, 1, 2)
plt.step(t_axis[:-1], u[:N_sim], where='post', label='u(t)')
plt.ylabel('Control'); plt.xlabel('Time step'); plt.legend(); plt.grid()
plt.tight_layout(); plt.show()
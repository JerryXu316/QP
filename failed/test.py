import numpy as np
import cvxpy as cp
import time
import pandas as pd
from multiprocessing import Pool

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
N_sim = 200
ref = 50.0

# Big-M参数（要比控制范围大）
M = 1000

# 正弦参考信号参数
A = 5          # 振幅
omega = 2 * np.pi / 120  # 角频率，周期 T = 120
C = 10           # 偏置
T = 120

# 初始化
def initialize_simulation():
    y = np.random.uniform(0, 20, N_sim + 1)  # y的初值在0-20之间
    u = np.zeros(N_sim + Np + delay + 1)
    phi = np.random.uniform(-T/2, T/2)  # 相位偏移量在-T/2到T/2之间
    C = 10  # 偏移量C保持不变
    return y, u, phi, C

# 单次仿真
def simulate_single_trajectory(seed):
    np.random.seed(seed)
    y, u, phi, C = initialize_simulation()
    data_records = []

    for t in range(N_sim):
        u_control = cp.Variable(Nc)
        delta_u = cp.Variable(Nc)
        z_pos = cp.Variable(Nc, boolean=True)
        z_neg = cp.Variable(Nc, boolean=True)

        y_pred = []
        prev_y = y[t]
        ref_t = np.array([A * np.sin(omega * (t + i + phi)) + C for i in range(1, Np + 1)])

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

        delta_u_expr = []
        for i in range(Nc):
            if i == 0:
                delta_u_expr.append(u_control[0] - (u[t-1] if t-1 >= 0 else 0))
            else:
                delta_u_expr.append(u_control[i] - u_control[i-1])
        delta_u_expr = cp.hstack(delta_u_expr)

        constraints = []
        for i in range(Nc):
            constraints.append(delta_u[i] == delta_u_expr[i])
            constraints.append(z_pos[i] + z_neg[i] <= 1)
            constraints.append(delta_u[i] >= 0.8 * z_pos[i] - M * (1 - z_pos[i]))
            constraints.append(delta_u[i] <= M * z_pos[i])
            constraints.append(delta_u[i] <= -0.8 * z_neg[i] + M * (1 - z_neg[i]))
            constraints.append(delta_u[i] >= -M * z_neg[i])

        cost = cp.quad_form(e, np.eye(Np)) + cp.quad_form(u_control, 0.01 * np.eye(Nc))
        prob = cp.Problem(cp.Minimize(cost), constraints)

        start_time = time.time()
        prob.solve(solver=cp.ECOS_BB, verbose=False)
        solve_time = time.time() - start_time

        if u_control.value is not None:
            u[t] = u_control.value[0]
            z_pos_val = np.round(z_pos.value).astype(int)
            z_neg_val = np.round(z_neg.value).astype(int)
        else:
            u[t] = 0
            z_pos_val = np.zeros(Nc, dtype=int)
            z_neg_val = np.zeros(Nc, dtype=int)

        u_delay_val = u[t - delay] if t - delay >= 0 else 0
        y[t+1] = a * y[t] + b * u_delay_val

        # === 保存数据 ===
        record = {
            't': t,
            'y_t': y[t],
            'u_tm1': u[t-1] if t > 0 else 0,
            **{f'z_pos_{i}': z_pos_val[i] for i in range(Nc)},
            **{f'z_neg_{i}': z_neg_val[i] for i in range(Nc)}
        }
        data_records.append(record)

        # 实时打印
        print(f"t={t}, time={solve_time:.3f}s, Δu取值数: {np.sum(z_pos_val) + np.sum(z_neg_val)}")

    return data_records

# 多线程并行仿真
def main():
    num_samples = 1000  # 仿真次数
    seeds = np.random.randint(0, 1000, num_samples)

    with Pool(processes=13) as pool:
        results = pool.map(simulate_single_trajectory, seeds)

    # 合并所有结果
    all_data_records = []
    for result in results:
        all_data_records.extend(result)

    # 保存为CSV文件
    df = pd.DataFrame(all_data_records)
    df.to_csv("miqp_sim_data.csv", index=False)
    print("数据保存完成：miqp_sim_data.csv")

if __name__ == "__main__":
    main()
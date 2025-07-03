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
b = K * (1 - a)

# MPC 参数
Np = 50
Nc = 10
delay = 15
N_sim = 200

# Big-M 参数
M = 1000

# 正弦参考信号参数
A = 5
omega = 2 * np.pi / 60  # 周期60

# 初始化函数，y=0
def initialize_simulation():
    y = np.zeros(N_sim + 1)      # 初始值固定0
    u = np.zeros(N_sim + Np + delay + 1)
    return y, u

# 单次仿真
def simulate_single_trajectory(seed):
    np.random.seed(seed)
    y, u = initialize_simulation()
    data_records = []

    for t in range(N_sim):
        u_control = cp.Variable(Nc)
        delta_u = cp.Variable(Nc)
        z_pos = cp.Variable(Nc, boolean=True)
        z_neg = cp.Variable(Nc, boolean=True)

        # 预测步的正弦参考信号
        ref_t = np.array([A * np.sin(omega * (t + i)) for i in range(1, Np + 1)])

        # 构建控制序列 u_full
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

        # y预测递推
        y_pred = []
        prev_y = y[t]
        for i in range(Np):
            y_next = a * prev_y + b * u_full[i]
            y_pred.append(y_next)
            prev_y = y_next

        y_pred_expr = cp.hstack(y_pred)
        e = y_pred_expr - ref_t

        # delta_u表达式
        delta_u_expr = []
        for i in range(Nc):
            if i == 0:
                delta_u_expr.append(u_control[0] - (u[t-1] if t-1 >= 0 else 0))
            else:
                delta_u_expr.append(u_control[i] - u_control[i-1])
        delta_u_expr = cp.hstack(delta_u_expr)

        # 约束
        constraints = []
        for i in range(Nc):
            constraints.append(delta_u[i] == delta_u_expr[i])
            constraints.append(z_pos[i] + z_neg[i] <= 1)
            constraints.append(delta_u[i] >= 0.8 * z_pos[i] - M * (1 - z_pos[i]))
            constraints.append(delta_u[i] <= M * z_pos[i])
            constraints.append(delta_u[i] <= -0.8 * z_neg[i] + M * (1 - z_neg[i]))
            constraints.append(delta_u[i] >= -M * z_neg[i])

        # 代价函数
        cost = cp.quad_form(e, np.eye(Np)) + cp.quad_form(u_control, 0.01 * np.eye(Nc))

        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        start_time = time.time()
        prob.solve(solver=cp.ECOS_BB, verbose=False)
        solve_time = time.time() - start_time

        # 获取结果
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

        # 保存数据
        record = {
            't': t,
            'y_t': y[t],
            'u_tm1': u[t-1] if t > 0 else 0,
            **{f'z_pos_{i}': z_pos_val[i] for i in range(Nc)},
            **{f'z_neg_{i}': z_neg_val[i] for i in range(Nc)}
        }
        data_records.append(record)

        print(f"t={t}, time={solve_time:.3f}s, Δu决策数: {np.sum(z_pos_val) + np.sum(z_neg_val)}")

    return data_records

# 并行多进程仿真
def main():
    num_samples = 1000  # 仿真轨迹数（可以改）
    num_cores = 13      # 并行核数

    seeds = np.random.randint(0, 10000, num_samples)

    with Pool(processes=num_cores) as pool:
        results = pool.map(simulate_single_trajectory, seeds)

    # 合并所有结果
    all_data_records = []
    for result in results:
        all_data_records.extend(result)

    # 保存csv
    df = pd.DataFrame(all_data_records)
    df.to_csv("miqp_sim_data_fixed.csv", index=False)
    print("✅ 数据保存完成：miqp_sim_data_fixed.csv")

if __name__ == "__main__":
    main()

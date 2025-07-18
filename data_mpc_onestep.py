import numpy as np
import cvxpy as cp
import time
import pandas as pd
from multiprocessing import Pool

# =================  系统参数（固定）  =================
Ts = 1
tau = 20
K = 0.8
a = np.exp(-Ts / tau)
b = K * tau * (1 - a)

# =================  MPC 参数（固定）  =================
Np     = 50
Nc     = 10
delay  = 15
N_sim  = 200
M      = 1000          # Big-M

# 正弦参考信号参数（固定）
A     = 5
omega = 2 * np.pi / 300
C     = 10
T_ref = 2 * np.pi / omega       # 周期 60

# 权重（固定）
Q  = np.eye(Np)
R0 = 0.01

# ----------------------------------------------------
# 以下函数与原始代码 100% 一致，只是抽出来方便并行
# ----------------------------------------------------
def initialize_simulation():
    y0   = np.random.uniform(0, 20)            # y 初值
    ubuf = np.zeros(delay)                # u 历史填零即可
    phi  = np.random.uniform(0, 2 * np.pi)     # 相位 0~2π
    return y0, ubuf, phi                       # ubuf 长度 = delay

def simulate_single_trajectory(seed):
    np.random.seed(seed)
    y0, u_hist, phi = initialize_simulation()
    # 初始化数组
    y = np.zeros(N_sim + 1)
    u = np.zeros(N_sim + delay + 1)
    y[0] = y0
    # 把 u_hist 填到负索引
    u[-delay:] = u_hist

    records = []

    for t in range(N_sim):
        # ------------ 原始 MPC 代码原封不动 ------------
        u_control = cp.Variable(Nc)
        delta_u   = cp.Variable(Nc)
        z_pos     = cp.Variable(Nc, boolean=True)
        z_neg     = cp.Variable(Nc, boolean=True)

        # 参考序列
        ref_t = np.array([A * np.sin(omega * (t + i + phi)) + C
                          for i in range(1, Np + 1)])

        # 构造 u_full
        u_full = []
        for i in range(Np):
            idx = t + i - delay
            if idx < 0:
                u_full.append(u[idx])  # 历史已存在
            elif 0 <= idx - t < Nc:
                u_full.append(u_control[idx - t])
            else:
                u_full.append(0.0)  # Nc 之后的步保持 0

        # 递推预测
        y_pred = []
        prev_y = y[t]
        for val in u_full:
            prev_y = a * prev_y + b * val
            y_pred.append(prev_y)
        y_pred_expr = cp.hstack(y_pred)
        e = y_pred_expr - ref_t

        # delta_u 表达式
        delta_u_expr = []
        for i in range(Nc):
            if i == 0:
                delta_u_expr.append(u_control[0] - u[t - 1])
            else:
                delta_u_expr.append(u_control[i] - u_control[i - 1])
        delta_u_expr = cp.hstack(delta_u_expr)

        # 约束
        constraints = []
        for i in range(Nc):
            constraints += [
                delta_u[i] == delta_u_expr[i],
                z_pos[i] + z_neg[i] <= 1,
                delta_u[i] >= 0.8 * z_pos[i] - M * (1 - z_pos[i]),
                delta_u[i] <=  M * z_pos[i],
                delta_u[i] <= -0.8 * z_neg[i] + M * (1 - z_neg[i]),
                delta_u[i] >= -M * z_neg[i]
            ]

        cost = cp.quad_form(e, Q) + cp.quad_form(u_control, R0 * np.eye(Nc))
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS_BB, verbose=False)

        # 取出实际控制
        if u_control.value is not None:
            u[t] = u_control.value[0]
            z_pos_val = np.round(z_pos.value).astype(int)
            z_neg_val = np.round(z_neg.value).astype(int)
        else:
            u[t] = 0
            z_pos_val = np.zeros(Nc, int)
            z_neg_val = np.zeros(Nc, int)

        # 系统更新
        y[t + 1] = a * y[t] + b * u[t - delay]

        # 记录
        rec = {
            'traj_id': seed,
            't': t,
            'phi': phi,
            'y0': y0,
            'y_t': y[t],
            'u_t': u[t],
            'delta_u_0': float(delta_u.value[0]) if delta_u.value is not None else 0,
            'solve_time': time.time() - (time.time())
        }
        for i in range(Nc):
            rec[f'z_pos_{i}'] = z_pos_val[i]
            rec[f'z_neg_{i}'] = z_neg_val[i]
        records.append(rec)

    return records

# ----------------------------------------------------
# 主函数：并行地毯式采样
# ----------------------------------------------------
def main():
    num_samples = 1000               # 先跑 1 万条试试
    seeds = np.random.randint(0, 100000, num_samples)

    # 并行
    with Pool(processes=13) as pool:    # 根据 CPU 核心数调整
        all_results = pool.map(simulate_single_trajectory, seeds)

    # 合并
    df = pd.DataFrame([row for traj in all_results for row in traj])
    # 1️⃣ 保存格式改成 CSV（无压缩）
    df.to_csv('carpet_miqp.csv', index=False)
    print('采样完成，数据已保存：carpet_miqp.csv')

    print('记录行数：', len(df))

if __name__ == "__main__":
    main()
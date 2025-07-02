import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import csv
import os

# ------------------ 系统和MPC参数 ------------------

class SystemParams:
    def __init__(self, Ts=1, tau=20, K=0.8, delay=15):
        self.Ts = Ts
        self.tau = tau
        self.K = K
        self.delay = delay
        self.a = np.exp(-Ts / tau)
        self.b = K * (1 - self.a)

class MPCParams:
    def __init__(self, Np=50, Nc=10, Q=None, R=None, big_M=1000):
        self.Np = Np  # 预测步长
        self.Nc = Nc  # 控制步长
        self.Q = Q if Q is not None else np.eye(Np)
        self.R = R if R is not None else 0.01 * np.eye(Nc)
        self.big_M = big_M

class ReferenceSignal:
    def __init__(self, A=5, omega=2*np.pi/60, C=10):
        self.A = A
        self.omega = omega
        self.C = C

    def get_ref(self, t, Np):
        # 返回时刻t的Np步参考信号数组
        return np.array([self.A * np.sin(self.omega * (t + i + 1)) + self.C for i in range(Np)])


# ------------------ 仿真核心函数 ------------------

def run_miqp_simulation(seed, system_params, mpc_params, ref_signal, N_sim=2500):
    np.random.seed(seed)

    a, b, delay = system_params.a, system_params.b, system_params.delay
    Np, Nc, Q, R, M = mpc_params.Np, mpc_params.Nc, mpc_params.Q, mpc_params.R, mpc_params.big_M

    y = np.zeros(N_sim + 1)
    u = np.zeros(N_sim + Np + delay + 1)

    data_records = []

    for t in range(N_sim):
        # 参考信号
        ref_t = ref_signal.get_ref(t, Np)

        # 定义变量
        u_control = cp.Variable(Nc)
        delta_u = cp.Variable(Nc)
        z_pos = cp.Variable(Nc, boolean=True)
        z_neg = cp.Variable(Nc, boolean=True)

        # 构造预测控制输入序列 u_full
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

        # 预测输出
        y_pred = []
        prev_y = y[t]
        for i in range(Np):
            y_next = a * prev_y + b * u_full[i]
            y_pred.append(y_next)
            prev_y = y_next
        y_pred_expr = cp.hstack(y_pred)

        e = y_pred_expr - ref_t

        # delta_u表达式构造
        delta_u_expr = []
        for i in range(Nc):
            if i == 0:
                delta_u_expr.append(u_control[0] - (u[t-1] if t > 0 else 0))
            else:
                delta_u_expr.append(u_control[i] - u_control[i-1])
        delta_u_expr = cp.hstack(delta_u_expr)

        constraints = []
        # delta_u定义约束
        for i in range(Nc):
            constraints.append(delta_u[i] == delta_u_expr[i])
            constraints.append(z_pos[i] + z_neg[i] <= 1)
            constraints.append(delta_u[i] >= 0.8 * z_pos[i] - M * (1 - z_pos[i]))
            constraints.append(delta_u[i] <= M * z_pos[i])
            constraints.append(delta_u[i] <= -0.8 * z_neg[i] + M * (1 - z_neg[i]))
            constraints.append(delta_u[i] >= -M * z_neg[i])

        cost = cp.quad_form(e, Q) + cp.quad_form(u_control, R)
        prob = cp.Problem(cp.Minimize(cost), constraints)

        start_time = time.time()
        prob.solve(solver=cp.ECOS_BB, verbose=False)
        solve_time = time.time() - start_time

        if u_control.value is not None:
            u[t] = u_control.value[0]
            delta_u_val = delta_u.value
            z_pos_val = z_pos.value
            z_neg_val = z_neg.value
        else:
            u[t] = 0
            delta_u_val = np.zeros(Nc)
            z_pos_val = np.zeros(Nc)
            z_neg_val = np.zeros(Nc)

        u_delay_val = u[t - delay] if t - delay >= 0 else 0
        y[t+1] = a * y[t] + b * u_delay_val

        # 记录数据
        data_records.append({
            't': t,
            'y': y[t],
            'u': u[t],
            'u_prev': u[t-1] if t > 0 else 0,
            'solve_time': solve_time,
            'delta_u': delta_u_val,
            'z_pos': z_pos_val,
            'z_neg': z_neg_val,
        })

    return data_records, y, u


# ------------------ 多线程调度 ------------------

def multiprocess_simulations(num_cores, system_params, mpc_params, ref_signal, N_sim=500):
    seeds = list(range(num_cores))
    args = [(seed, system_params, mpc_params, ref_signal, N_sim) for seed in seeds]

    with mp.Pool(num_cores) as pool:
        results = pool.starmap(run_miqp_simulation, args)

    # 合并所有数据
    all_data = []
    for data_records, _, _ in results:
        all_data.extend(data_records)

    return all_data


# ------------------ 数据保存 ------------------

def save_data_to_csv(data, filename='miqp_simulation_data.csv'):
    # data是字典列表，delta_u,z_pos,z_neg是数组，存储为字符串
    with open(filename, mode='w', newline='') as f:
        fieldnames = ['t', 'y', 'u', 'u_prev', 'solve_time', 'delta_u', 'z_pos', 'z_neg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            row_copy = row.copy()
            row_copy['delta_u'] = ','.join(map(str, row['delta_u']))
            row_copy['z_pos'] = ','.join(map(str, row['z_pos']))
            row_copy['z_neg'] = ','.join(map(str, row['z_neg']))
            writer.writerow(row_copy)


# ------------------ 绘图 ------------------

def plot_example(y, u, ref_signal, N_sim=500):
    t_axis = np.arange(N_sim)
    ref_vals = np.array([ref_signal.get_ref(t, 1)[0] for t in t_axis])

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_axis, y[:N_sim], label='Output y')
    plt.plot(t_axis, ref_vals, '--', label='Reference signal')
    plt.ylabel('Output y')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.step(t_axis, u[:N_sim], where='post', label='Control u')
    plt.ylabel('Control u')
    plt.xlabel('Time step')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# ------------------ 主程序 ------------------

if __name__ == '__main__':
    # 初始化参数
    system_params = SystemParams()
    mpc_params = MPCParams()
    ref_signal = ReferenceSignal()

    # 多线程仿真
    num_cores = 8   # 你可以改为16，根据CPU核数
    print(f"Start {num_cores} parallel MIQP simulations...")
    all_sim_data = multiprocess_simulations(num_cores, system_params, mpc_params, ref_signal, N_sim=500)
    print(f"Simulation finished. Total samples collected: {len(all_sim_data)}")

    # 保存数据
    save_data_to_csv(all_sim_data, 'miqp_data.csv')
    print("Data saved to miqp_data.csv")
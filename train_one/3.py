import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import gc
from tqdm import tqdm
import csv

# 可外部修改的参数
K_model = 0.8
T_model = 20
K_actual = 0.85
T_actual = 22
Ts = 1
D = 5
P = 20
M = 10
total_steps = 200
external_disturbances = 0.01

# 控制输入约束
y_min = 0.0
y_max = 2.0
r_min = 0.0
r_max = 2.0
u_min = -2.0
u_max = 2.0
delta_u_min = -1.0
delta_u_max = 1.0
positive_delta_u_constrain = 0.4
negative_delta_u_constrain = -0.4

# 训练集参数
Q = np.diag(np.ones(P) * 1.0)
R = np.diag(np.ones(M) * 0.00)
R_delta = np.diag(np.ones(M) * 0.001)

# 离散化系数
a_model = np.exp(-Ts / T_model)
b0_model = K_model * (1 - a_model)
a_actual = np.exp(-Ts / T_actual)
b0_actual = K_actual * (1 - a_actual)

# 状态矩阵
n = D + 1
A_model = np.zeros((n, n))
A_model[0, 0] = a_model
A_model[0, -1] = b0_model
for i in range(2, n):
    A_model[i, i - 1] = 1

b = np.zeros((n, 1))
b[1, 0] = 1
c = np.zeros((1, n))
c[0, 0] = 1

C = np.zeros(n)
C[0] = 1

# Fx 和 Gx
Fx = np.zeros((n * P, n))
Gx = np.zeros((n * P, M))
for i in range(P):
    Fx[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A_model, i + 1)
    for j in range(min(i + 1, M)):
        Gx[i * n:(i + 1) * n, j] = (np.linalg.matrix_power(A_model, i - j) @ b).flatten()
    if i >= M:
        Gx[i * n:(i + 1) * n, M - 1] = sum((np.linalg.matrix_power(A_model, j) @ b).flatten() for j in range(i - M + 1))

# Fy 和 Gy
Fy = np.zeros((P, n))
Gy = np.zeros((P, M))
for i in range(P):
    Fy[i, :] = c @ np.linalg.matrix_power(A_model, i + 1)
for i in range(P):
    for j in range(min(i + 1, M)):
        Gy[i, j] = (c @ np.linalg.matrix_power(A_model, i - j) @ b).item()
    if i >= M:
        Gy[i, M - 1] = sum((c @ np.linalg.matrix_power(A_model, j) @ b).item() for j in range(i - M + 1))

def setup_model():
    model = gp.Model()
    model.Params.OutputFlag = 0  # 静默

    U = model.addMVar(M, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="U")
    err = model.addMVar(P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="err")
    p = model.addMVar(M, vtype=GRB.BINARY, name="p")
    q = model.addMVar(M, vtype=GRB.BINARY, name="q")
    delta_u = model.addMVar(M, lb=delta_u_min, ub=delta_u_max, name="delta_u")

    return model, U, err, p, q, delta_u

def update_model(model, U, err, p, q, delta_u, y_k, r_k, x_model):
    y_offset = Fy @ x_model.T
    for i in range(P):
        model.addConstr(err[i] == y_offset[i] + Gy[i, :] @ U - r_k)

    model.addConstr(delta_u[0] == U[0] - x_model[-1])
    for i in range(1, M):
        model.addConstr(delta_u[i] == U[i] - U[i - 1])

    for i in range(M):
        model.addConstr(p[i] + q[i] <= 1)
        model.addConstr(delta_u[i] <= negative_delta_u_constrain * p[i] + delta_u_max * (1 - p[i]))
        model.addConstr(delta_u[i] >= positive_delta_u_constrain * q[i] + delta_u_min * (1 - q[i]))
        model.addConstr(delta_u[i] <= delta_u_max * q[i])
        model.addConstr(delta_u[i] >= delta_u_min * p[i])

    cost = gp.QuadExpr()
    cost += err @ Q @ err
    cost += U @ R @ U
    cost += delta_u @ R_delta @ delta_u
    model.setObjective(cost, GRB.MINIMIZE)

def miqp_solve(data_row):
    y_k = data_row[0]
    r_k = data_row[D + 1]
    x_model = np.zeros(n)
    x_model[0] = y_k
    x_model[1 : D + 1] = data_row[1:D + 1]

    model, U, err, p, q, delta_u = setup_model()
    update_model(model, U, err, p, q, delta_u, y_k, r_k, x_model)
    model.optimize()

    # 检查是否找到最优解
    if model.status != GRB.OPTIMAL:
        model.dispose()
        return [0] * M  # 退回默认值

    p_optimal = p.X
    q_optimal = q.X

    # 检查 p 和 q 是否在合理范围内
    if not np.all(np.isin(p_optimal, [0, 1])) or not np.all(np.isin(q_optimal, [0, 1])):
        model.dispose()
        return [0] * M  # 若p或q不合法，返回默认值

    # 生成 t 值
    t_values = []
    for i in range(M):
        if p_optimal[i] == 1 and q_optimal[i] == 0:
            t_values.append(-1)
        elif p_optimal[i] == 0 and q_optimal[i] == 0:
            t_values.append(0)
        elif p_optimal[i] == 0 and q_optimal[i] == 1:
            t_values.append(1)

    model.dispose()  # 显式释放模型资源
    gc.collect()

    return t_values

def generate_and_save_dataset_streaming(input_csv, output_csv, flush_every=50):
    data = pd.read_csv(input_csv)
    total = len(data)

    column_names = ['y(k)', 'u(k-1)', 'u(k-2)', 'u(k-3)','u(k-4)','u(k-5)', 'r(k)'] + [f't(k+{i})' for i in range(M)]
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)

        buffer = []
        for idx, row in tqdm(data.iterrows(), total=total, desc="Solving and Writing"):
            try:
                input_data = row.tolist()
                t_values = miqp_solve(row)
                buffer.append(input_data + t_values)
            except Exception as e:
                print(f"[ERROR] Row {idx}: {e}")
                continue

            if len(buffer) >= flush_every:
                writer.writerows(buffer)
                f.flush()
                buffer.clear()
                gc.collect()

        if buffer:
            writer.writerows(buffer)
            f.flush()
            buffer.clear()
            gc.collect()

    print(f"✅ 数据集生成完成：{output_csv}")

if __name__ == "__main__":
    generate_and_save_dataset_streaming("./input/32.csv", "./output/32.csv", flush_every=50)
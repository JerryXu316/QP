import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd


# 可外部修改的参数
K_model = 0.8  # 模型增益
T_model = 20  # 模型时间常数
K_actual = 0.85  # 实际增益
T_actual = 22  # 实际时间常数
Ts = 1  # 离散化步长
D = 3  # 历史控制输入维度
P = 20  # 预测时域
M = 5  # 控制时域
total_steps = 200  # 总时间步长
external_disturbances = 0.01  # 外部扰动

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
Q = np.diag(np.ones(P) * 1.0)  # err 的权重矩阵
R = np.diag(np.ones(M) * 0.00)  # U 的权重矩阵
R_delta = np.diag(np.ones(M) * 0.001)  # delta_u 的权重矩阵


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

A_actual = np.zeros((n, n))
A_actual[0, 0] = a_actual
A_actual[0, -1] = b0_actual
for i in range(2, n):
    A_actual[i, i - 1] = 1

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




def load_csv_data(file_path):
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    return data


def miqp_solve(data_row):

    y_k = data_row[0]
    r_k = data_row[D + 1]  # r(k)

    x_model = np.zeros(n)
    x_model[0] = y_k
    x_model[1 : D + 1] = data_row[1:D + 1]


    # 用读取的数据初始化模型并求解
    model = gp.Model()

    # 控制变量 U
    U = model.addMVar(M, lb=u_min, ub=u_max, vtype=GRB.CONTINUOUS, name="U")

    # 误差变量 err
    err = model.addMVar(P, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="err")

    # 二进制变量 p 和 q
    p = model.addMVar(M, vtype=GRB.BINARY, name="p")
    q = model.addMVar(M, vtype=GRB.BINARY, name="q")

    # 计算 y_offset
    y_offset = Fy @ x_model.T

    # 添加 err 的约束
    for i in range(P):
        model.addConstr(err[i] == y_offset[i] + Gy[i, :] @ U - r_k)

    # delta_u 的约束
    delta_u = model.addMVar(M, lb=delta_u_min, ub=delta_u_max, name="delta_u")
    model.addConstr(delta_u[0] == U[0] - x_model[-1])
    for i in range(1, M):
        model.addConstr(delta_u[i] == U[i] - U[i - 1])

    # 添加 p 和 q 的约束
    for i in range(M):
        model.addConstr(p[i] + q[i] <= 1)
        model.addConstr(delta_u[i] <= negative_delta_u_constrain * p[i] + delta_u_max * (1 - p[i]))
        model.addConstr(delta_u[i] >= positive_delta_u_constrain * q[i] + delta_u_min * (1 - q[i]))
        model.addConstr(delta_u[i] <= delta_u_max * q[i])
        model.addConstr(delta_u[i] >= delta_u_min * p[i])

    # 代价函数
    cost = gp.QuadExpr()
    cost += err @ Q @ err
    cost += U @ R @ U
    cost += delta_u @ R_delta @ delta_u
    model.setObjective(cost, GRB.MINIMIZE)

    # 求解
    model.optimize()
    u_optimal = U.X
    p_optimal = p.X
    q_optimal = q.X

    # 根据 (p, q) 的选择生成 t 的值
    t_values = []
    for i in range(M):
        if p_optimal[i] == 1 and q_optimal[i] == 0:
            t_values.append(-1)
        elif p_optimal[i] == 0 and q_optimal[i] == 0:
            t_values.append(0)
        elif p_optimal[i] == 0 and q_optimal[i] == 1:
            t_values.append(1)

    return t_values


def create_dataset_from_csv(csv_file_path):
    data = load_csv_data(csv_file_path)
    dataset = []

    for index, row in data.iterrows():
        # 使用 CSV 中的每一行作为输入数据，获取相应的 t 值
        t_values = miqp_solve(row)

        # 将输入参数和 t 值组合成一个新的样本，作为神经网络的训练数据
        input_data = row.tolist()
        output_data = np.array(t_values)  # 只包含 t 的值
        dataset.append([*input_data, *output_data])

    return np.array(dataset)


def save_dataset_to_csv(dataset, output_file_path):
    # 列标签
    column_names = ['y(k)', 'u(k-1)', 'u(k-2)', 'u(k-3)', 'r(k)', 't(k)', 't(k+1)', 't(k+2)', 't(k+3)', 't(k+4)']

    # 将数据集转换为 pandas DataFrame
    df = pd.DataFrame(dataset, columns=column_names)

    # 保存为 CSV 文件
    df.to_csv(output_file_path, index=False)

    print(f"Dataset saved to {output_file_path}")


def generate_and_save_dataset(input_csv, output_csv):
    dataset = create_dataset_from_csv(input_csv)
    save_dataset_to_csv(dataset, output_csv)

# 调用生成数据集的函数
generate_and_save_dataset("initial_generated_data.csv", "solve_generated_data.csv")

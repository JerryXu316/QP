import numpy as np
import pandas as pd
from pyDOE import lhs

# 数据集大小
total_samples = 1000000

# 可外部修改的参数
K_model = 0.8  # 模型增益
T_model = 20  # 模型时间常数
K_actual = 0.85  # 实际增益
T_actual = 22  # 实际时间常数
Ts = 1  # 离散化步长
D = 3  # 历史控制输入维度
P = 20  # 预测时域
M = 5  # 控制时域
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


# 生成数据集并保存为CSV文件的函数
def generate_and_save_dataset(file_name='dataset.csv'):
    dataset = []

    for k in range(total_samples):
        # 使用LHS采样生成 delta_u(k-1), ..., delta_u(k-D+1), u(k-D), r(k), y(k)
        lhs_sample = lhs(D + 2, samples=1)  # D-1个delta_u维度 + 1个u(k-D)维度 + 1个r维度 + 1个y维度

        # delta_u_samples的维度前 D-1 个是 delta_u(k-1), ..., delta_u(k-D+1)
        delta_u_samples = delta_u_min + (delta_u_max - delta_u_min) * lhs_sample.flatten()[:D - 1]  # 映射到 delta_u 范围

        # 后3个维度分别对应 u(k-D), r(k), y(k)
        u_k_D = u_min + (u_max - u_min) * lhs_sample[0, D - 1]  # 映射到u(k-D)的范围
        r_k = r_min + (r_max - r_min) * lhs_sample[0, D]  # 映射到r(k)的范围
        y_k = y_min + (y_max - y_min) * lhs_sample[0, D + 1]  # 映射到y(k)的范围

        # 将LHS采样的结果进行映射到三个区间：[delta_u_min, negative_delta_u_constrain], {0}, [positive_delta_u_constrain, delta_u_max]
        for i in range(D - 1):
            if delta_u_samples[i] < negative_delta_u_constrain:
                delta_u_samples[i] = np.random.uniform(delta_u_min, negative_delta_u_constrain)
            elif delta_u_samples[i] > positive_delta_u_constrain:
                delta_u_samples[i] = np.random.uniform(positive_delta_u_constrain, delta_u_max)
            else:
                delta_u_samples[i] = 0

        # 根据delta_u计算u(k-1), ..., u(k-D)
        u_history = np.zeros(D)
        u_history[0] = np.random.uniform(u_min, u_max)  # 初始控制输入随机生成
        for i in range(1, D):
            u_history[i] = u_history[i - 1] + delta_u_samples[i - 1]
            u_history[i] = np.clip(u_history[i], u_min, u_max)  # 确保 u(k-1), ..., u(k-D) 在控制输入的范围内

        # 将生成的数据点加入到数据集中
        dataset.append([y_k, *u_history, r_k])

    # 将数据保存到 CSV 文件
    df = pd.DataFrame(dataset, columns=['y(k)', 'u(k-1)', 'u(k-2)', 'u(k-3)', 'r(k)'])
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")


# 调用函数生成数据集并保存到文件
generate_and_save_dataset('initial_generated_data.csv')

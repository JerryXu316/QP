import numpy as np
import control

# ======================
# 1. 定义连续时间系统
# ======================

# 定义连续状态空间模型
# 状态矩阵 A_cont
A_cont = np.array([[0, 1],
                   [-2, -3]])

# 输入矩阵 B_cont
B_cont = np.array([[0],
                   [1]])

# 输出矩阵 C_cont
C_cont = np.array([[1, 1]])

# 直接馈通矩阵 D_cont
D_cont = 0

# 创建连续时间状态空间系统
sys_cont = control.ss(A_cont, B_cont, C_cont, D_cont)

# ======================
# 2. 离散化系统
# ======================

# 定义采样周期 (单位：秒)
Ts = 0.1

# 选择离散化方法
# 'zoh'：零阶保持 (Zero-Order Hold)，最常用
method = 'zoh'

# 进行离散化
sys_disc = control.c2d(sys_cont, Ts, method=method)

# ======================
# 3. 提取离散化后的状态空间矩阵
# ======================

# 离散状态矩阵 A_d
Ad = sys_disc.A

# 离散输入矩阵 B_d
Bd = sys_disc.B

# 离散输出矩阵 C_d
Cd = sys_disc.C

# 离散前馈矩阵 D_d
Dd = sys_disc.D

# ======================
# 4. 显示离散化结果
# ======================

print("===== 离散化后的状态空间模型 =====")
print(f"离散状态矩阵 A_d:\n{Ad}")
print(f"离散输入矩阵 B_d:\n{Bd}")
print(f"离散输出矩阵 C_d:\n{Cd}")
print(f"离散前馈矩阵 D_d:\n{Dd}")

# ======================
# 5. 系统特性分析（可选）
# ======================

# 计算离散系统的极点
poles_disc = control.poles(sys_disc)
print(f"离散系统极点: {poles_disc}")

# 检查离散系统是否稳定（所有极点模是否小于1）
is_stable = all(np.abs(poles_disc) < 1)
print(f"离散系统是否稳定: {is_stable}")

# ======================
# 6. 验证离散化结果（可选）
# ======================

# 定义初始状态和输入
x_k = np.array([[1], [0]])  # 初始状态 x[0]
u_k = np.array([[1]])       # 输入 u[0]

# 按照离散状态方程计算 x[1] = A_d x[0] + B_d u[0]
x_k_plus_1 = Ad @ x_k + Bd @ u_k

print("\n验证离散状态方程:")
print(f"给定 x[0] = \n{x_k.flatten()}")
print(f"u[0] = {u_k.flatten()[0]}")
print(f"计算 x[1] = A_d x[0] + B_d u[0] = \n{x_k_plus_1.flatten()}")

# ======================
# 7. 可视化（可选）
# ======================

# 绘制阶跃响应比较（连续 vs 离散）
import matplotlib.pyplot as plt

# 定义时间向量
t_cont = np.linspace(0, 1, 1000)  # 连续时间
t_disc = np.arange(0, 1 + Ts, Ts)  # 离散时间点

# 计算连续系统的阶跃响应
t_cont, y_cont = control.step_response(sys_cont, T=t_cont)

# 计算离散系统的阶跃响应
t_disc, y_disc = control.step_response(sys_disc, T=t_disc)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(t_cont, y_cont, 'b-', label='连续系统')
plt.step(t_disc, y_disc, 'r--', where='post', label='离散系统 (ZOH)')
plt.xlabel('时间 (秒)')
plt.ylabel('输出')
plt.title('连续 vs 离散系统阶跃响应')
plt.legend()
plt.grid(True)
plt.show()
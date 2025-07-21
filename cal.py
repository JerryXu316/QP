from simple_gurobi import solve_miqp_gurobi

# 构造你的优化参数（Q, c, A, b等），例如
import numpy as np
Q = np.array([[2, 0], [0, 2]])
c = np.array([1, 1])
A = np.array([[1, 2]])
b = np.array([4])

# 调用求解函数
result = solve_miqp_gurobi(Q=Q, c=c, A=A, b=b)
print(result)
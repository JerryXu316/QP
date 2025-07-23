import gurobipy as gp
from gurobipy import GRB

model = gp.Model("miqp_demo")

# 变量
x = model.addVar(vtype=GRB.CONTINUOUS, name='x')
y = model.addVar(vtype=GRB.BINARY, name='y')

# 约束
model.addConstr(x + y == 1)

# 二次目标
obj = x * x + 2 * x * y + y  # x^2 + 2xy + y
model.setObjective(obj, GRB.MINIMIZE)

model.optimize()

if model.status == GRB.OPTIMAL:
    print("x =", x.X)
    print("y =", y.X)
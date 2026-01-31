import numpy as np

# 1. 定义系数矩阵 A (左边的方阵)
A = np.array([
    [2, 5],
    [1, 3]
])

# 2. 定义目标向量 b (等号右边的向量)
b = np.array([1,2])

# 3. 求解方程 Ax = b
S = A @ b

print(f"解得 (x, y, z) 为: {S}")
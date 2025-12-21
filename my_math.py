import numpy as np
import matplotlib.pyplot as plt

# 训练数据：身高 (x) 和体重 (y)
X = np.array([170, 180, 160, 190, 165])  # 身高数据
y = np.array([65, 75, 55, 85, 60])       # 体重数据

# 将身高数据转换为矩阵形式，并添加偏差项 (1列)
X_b = np.c_[np.ones((len(X), 1)), X]  # 添加偏差项列 (每个身高前加 1)

# 计算最佳参数 (w1 和 b)，使用正规方程
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 提取出权重 w1 和偏差项 b
b = theta_best[0]
w1 = theta_best[1]

print(f"最佳偏差项 b: {b}")
print(f"最佳权重 w1: {w1}")

# 预测：给定身高 175cm，计算预测体重
X_new = np.array([[1, 175]])  # 175cm的身高，添加偏差项 1
y_predict = X_new.dot(theta_best)  # 预测体重
print(f"预测身高 175cm 的体重: {y_predict[0]} kg")

# 可视化结果
plt.scatter(X, y, color='blue', label='实际数据')  # 绘制实际数据点
plt.plot(X, X_b.dot(theta_best), color='red', label='拟合直线')  # 绘制拟合直线
plt.xlabel('身高 (cm)')
plt.ylabel('体重 (kg)')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 线性回归
from sklearn.preprocessing import PolynomialFeatures # 多项式特征
from sklearn.model_selection import train_test_split # 分割数据集
from sklearn.metrics import mean_squared_error # 均方误差

# ```
# 1. 生成数据
# 2. 划分训练集和测试集
# 3. 定义模型（线性回归）
# 4. 训练模型
# 5. 预测结果，计算损失
# ```

# 生成数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

# 画出散点图（3个子图）
fig , ax= plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c='y')
ax[1].scatter(X, y, c='y')
ax[2].scatter(X, y, c='y')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型（线性回归）
model = LinearRegression()

# 1. 欠拟合
X_train1 = X_train
X_test1 = X_test
model.fit(X_train1, y_train)

## 打印查看模型参数
print("=== 欠拟合模型参数 ===")
print("系数：", model.coef_)
print("截距：", model.intercept_)

## 预测结果，计算误差
y_pred = model.predict(X_test1)
test_loss1 = mean_squared_error(y_test, y_pred)
train_loss1 = mean_squared_error(y_train, model.predict(X_train1))

## 画出拟合曲线，写出训练误差和测试误差
ax[0].plot(X,model.predict(X), c='r')
ax[0].text(-3,1,f"测试误差：{test_loss1:.4f}")
ax[0].text(-3,1.3,f"训练误差：{train_loss1:.4f}")

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


# 2. 恰好拟合
poly5=PolynomialFeatures(degree=5)
X_train2 = poly5.fit_transform(X_train1)
X_test2 = poly5.fit_transform(X_test1)

## 训练模型
model.fit(X_train2, y_train)

## 打印查看模型参数
print("\n=== degree=5 模型参数 ===")
print("系数：", model.coef_)
print("截距：", model.intercept_)

## 预测结果，计算误差
y_pred = model.predict(X_test2)
test_loss2 = mean_squared_error(y_test, y_pred)
train_loss2 = mean_squared_error(y_train, model.predict(X_train2))

## 画出拟合曲线，写出训练误差和测试误差
ax[1].plot(X,model.predict(poly5.fit_transform(X)), c='r')
ax[1].text(-3,1,f"测试误差：{test_loss2:.4f}")
ax[1].text(-3,1.3,f"训练误差：{train_loss2:.4f}")

# 3. 过拟合
poly20=PolynomialFeatures(degree=20)
X_train3 = poly20.fit_transform(X_train1)
X_test3 = poly20.fit_transform(X_test1)
model.fit(X_train3, y_train)

## 打印查看模型参数
print("\n=== degree=20 模型参数 ===")
print("系数：", model.coef_)
print("截距：", model.intercept_)

y_pred = model.predict(X_test3)
test_loss3 = mean_squared_error(y_test, y_pred)
train_loss3 = mean_squared_error(y_train, model.predict(X_train3))
ax[2].plot(X,model.predict(poly20.fit_transform(X)), c='r')
ax[2].text(-3,1,f"测试误差：{test_loss3:.4f}")
ax[2].text(-3,1.3,f"训练误差：{train_loss3:.4f}")

plt.show()


















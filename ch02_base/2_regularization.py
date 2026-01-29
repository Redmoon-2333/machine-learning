import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型，Lasso回归，岭回归
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.metrics import mean_squared_error  # 均方误差损失函数


plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


# 生成数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

# 画出散点图（2*3个子图）
fig , ax= plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].scatter(X, y, c='y')
ax[0,1].scatter(X, y, c='y')
ax[0,2].scatter(X, y, c='y')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly20=PolynomialFeatures(degree=20)
X_train = poly20.fit_transform(X_train)
X_test = poly20.transform(X_test)

# 1. 不加正则化项，过拟合
## 定义模型
model = LinearRegression()

## 训练模型
model.fit(X_train, y_train)

## 打印查看模型参数
y_pred1 = model.predict(X_test)
test_loss1 = mean_squared_error(y_test, y_pred1)

ax[0,0].plot(X,model.predict(poly20.fit_transform(X)), c='r')
ax[0,0].text(-3,1,f"测试误差：{test_loss1:.4f}")
## 画所有系数的直方图
ax[1,0].bar(np.arange(21), model.coef_.reshape(-1))

# 2. 加L1正则化项（Lasso回归）
## 定义模型
lasso = Lasso(alpha=0.01)

## 训练模型
lasso.fit(X_train, y_train)

## 打印查看模型参数
y_pred2 = lasso.predict(X_test)
test_loss2 = mean_squared_error(y_test, y_pred2)

ax[0,1].plot(X,lasso.predict(poly20.fit_transform(X)), c='r')
ax[0,1].text(-3,1,f"测试误差：{test_loss2:.4f}")
## 画所有系数的直方图
ax[1,1].bar(np.arange(21), lasso.coef_.reshape(-1))

# 3. 加L2正则化项（岭回归）
## 定义模型
ridge = Ridge(alpha=1)

## 训练模型
ridge.fit(X_train, y_train)

## 打印查看模型参数
y_pred3 = ridge.predict(X_test)
test_loss3 = mean_squared_error(y_test, y_pred3)

ax[0,2].plot(X,ridge.predict(poly20.fit_transform(X)), c='r')
ax[0,2].text(-3,1,f"测试误差：{test_loss3:.4f}")
## 画所有系数的直方图
ax[1,2].bar(np.arange(21), ridge.coef_.reshape(-1))
plt.show()


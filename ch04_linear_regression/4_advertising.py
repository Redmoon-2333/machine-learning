
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# 1. 读取数据
data = pd.read_csv('../data/Advertising1.csv')

# 2. 数据预处理
data.dropna(inplace=True)
data.drop(columns=data.columns[0], inplace=True)

print(data.head())

# 3. 划分训练集和测试集
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 4. 特征工程：标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 创建模型并训练
# 5.1 正规方程法
model_lr=LinearRegression()
model_lr.fit(X_train_scaled, y_train)

print("LR Coefficient: ", model_lr.coef_)
print("LR Intercept: ", model_lr.intercept_)

# 5.2 SGD
model_sgd=SGDRegressor()
model_sgd.fit(X_train_scaled, y_train)
print("SGD Coefficient: ", model_sgd.coef_)
print("SGD Intercept: ", model_sgd.intercept_)

# 6 预测
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_sgd = model_sgd.predict(X_test_scaled)

# 7 使用均方误差评价模型
print("LR Mean Squared Error: ", mean_squared_error(y_test, y_pred_lr))
print("SGD Mean Squared Error: ", mean_squared_error(y_test, y_pred_sgd))





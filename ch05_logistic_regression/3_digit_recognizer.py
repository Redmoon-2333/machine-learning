
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# 1.加载数据集
data = pd.read_csv('../data/train.csv')

# # 测试图像
# test_image = data.iloc[10,1:] . values
# plt.imshow(test_image.reshape(28,28))
# plt.show()

# 2. 划分数据集
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程：归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 定义模型和训练
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 5. 模型评估
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# 6. 测试
digits = X_test[123,:].reshape(1,-1)
print(model.predict(digits))
print(y_test.iloc[123])
print(model.predict_proba(digits))

# 7. 画出图形
plt.imshow(digits.reshape(28,28))
plt.show()

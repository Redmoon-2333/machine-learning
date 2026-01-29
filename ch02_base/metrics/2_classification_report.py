from sklearn.datasets import make_classification  # 自动生成分类数据集
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类模型
from sklearn.metrics import classification_report  # 生成分类评估报告

# 1. 生成数据
X, y =make_classification(n_samples=1000, n_features=20, n_classes=2,random_state=42)
#print(X.shape, y.shape)

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3. 定义一个分类模型
model = LogisticRegression()
# 4. 训练模型
model.fit(X_train, y_train)
# 5. 预测，生成报告
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))






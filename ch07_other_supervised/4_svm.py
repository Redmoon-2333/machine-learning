# 支持向量机(SVM)分类示例
from sklearn.svm import SVC
from sklearn.datasets import load_iris, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 1. 线性SVM
print("=" * 50)
print("线性SVM - 鸢尾花分类")
print("=" * 50)

iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化（SVM对特征尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性核SVM
linear_svm = SVC(kernel='linear', C=1.0, random_state=42)
linear_svm.fit(X_train_scaled, y_train)
y_pred = linear_svm.predict(X_test_scaled)
print(f"线性SVM准确率: {accuracy_score(y_test, y_pred):.4f}")


# 2. 非线性SVM（RBF核）
print("\n" + "=" * 50)
print("非线性SVM(RBF核) - 月牙形数据")
print("=" * 50)

# 生成非线性数据
X_moon, y_moon = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moon, y_moon, test_size=0.3, random_state=42)

# RBF核SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
rbf_svm.fit(X_train_m, y_train_m)
y_pred_m = rbf_svm.predict(X_test_m)
print(f"RBF核SVM准确率: {accuracy_score(y_test_m, y_pred_m):.4f}")


# 3. 不同核函数对比
print("\n" + "=" * 50)
print("不同核函数对比")
print("=" * 50)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    score = svm.score(X_test_scaled, y_test)
    print(f"{kernel:10s}核准确率: {score:.4f}")


# 4. 可视化决策边界（使用2个特征）
print("\n绘制决策边界...")
X_2d = X[:, :2]  # 只用前两个特征
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.3, random_state=42)

scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)

svm_2d = SVC(kernel='rbf', C=1.0)
svm_2d.fit(X_train_2d_scaled, y_train_2d)

# 创建网格
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], c=y_train_2d, cmap='viridis', edgecolors='black')
plt.xlabel('特征1 (标准化)')
plt.ylabel('特征2 (标准化)')
plt.title('SVM决策边界可视化')
plt.show()

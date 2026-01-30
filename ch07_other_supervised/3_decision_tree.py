# 决策树分类示例
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树模型
model = DecisionTreeClassifier(
    criterion='gini',      # 分裂标准: 'gini'(基尼指数) 或 'entropy'(信息增益)
    max_depth=3,           # 最大深度，防止过拟合
    min_samples_split=5,   # 节点分裂所需最小样本数
    min_samples_leaf=2,    # 叶节点最小样本数
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

# 查看特征重要性
print("\n特征重要性:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# 可视化决策树
plt.figure(figsize=(15, 10))
plot_tree(model, 
          feature_names=feature_names, 
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('决策树可视化')
plt.tight_layout()
plt.show()


# 使用信息增益(entropy)作为分裂标准
print("\n" + "=" * 50)
print("使用信息增益(entropy)的决策树")
print("=" * 50)

model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred_entropy):.4f}")

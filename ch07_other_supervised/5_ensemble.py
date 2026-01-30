# 集成学习示例（AdaBoost和随机森林）
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 1. AdaBoost
print("=" * 50)
print("AdaBoost 集成学习")
print("=" * 50)

# 创建AdaBoost模型（基分类器为决策树桩）
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 弱分类器：决策树桩
    n_estimators=50,      # 弱分类器数量
    learning_rate=1.0,    # 学习率
    random_state=42
)

ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print(f"AdaBoost准确率: {accuracy_score(y_test, y_pred_ada):.4f}")


# 2. 随机森林
print("\n" + "=" * 50)
print("随机森林 集成学习")
print("=" * 50)

# 创建随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,      # 决策树数量
    max_depth=5,           # 最大深度
    max_features='sqrt',   # 每次分裂随机选择的特征数
    min_samples_split=5,   # 节点分裂所需最小样本数
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"随机森林准确率: {accuracy_score(y_test, y_pred_rf):.4f}")

# 特征重要性
print("\n特征重要性:")
for name, importance in zip(feature_names, rf_model.feature_importances_):
    print(f"  {name}: {importance:.4f}")


# 3. 对比单棵决策树和集成方法
print("\n" + "=" * 50)
print("单棵决策树 vs 集成方法对比")
print("=" * 50)

# 单棵决策树
single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
single_tree.fit(X_train, y_train)
print(f"单棵决策树准确率: {single_tree.score(X_test, y_test):.4f}")
print(f"AdaBoost准确率:   {accuracy_score(y_test, y_pred_ada):.4f}")
print(f"随机森林准确率:   {accuracy_score(y_test, y_pred_rf):.4f}")


# 4. 可视化特征重要性
plt.figure(figsize=(10, 5))
indices = np.argsort(rf_model.feature_importances_)[::-1]
plt.bar(range(len(feature_names)), rf_model.feature_importances_[indices])
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()


# 5. 不同树数量对随机森林性能的影响
print("\n" + "=" * 50)
print("树数量对随机森林性能的影响")
print("=" * 50)

n_trees_list = [1, 5, 10, 50, 100, 200]
scores = []

for n_trees in n_trees_list:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    scores.append(score)
    print(f"n_estimators={n_trees:3d}: 准确率={score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(n_trees_list, scores, 'bo-')
plt.xlabel('决策树数量')
plt.ylabel('准确率')
plt.title('树数量对随机森林性能的影响')
plt.grid(True)
plt.show()

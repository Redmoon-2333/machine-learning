# KNN算法技术文档

## 1. KNN算法介绍

KNN（K-Nearest Neighbors，K近邻算法）是一种经典且直观的机器学习算法，既可以用于分类任务，也可以用于回归任务。KNN算法的核心思想是"物以类聚"，即相似的数据点往往具有相似的标签或属性值。

### 1.1 工作原理

KNN算法是一种基于实例的学习方法，也称为懒惰学习（Lazy Learning）。它的工作原理可以概括为以下几个步骤：

**基本思想**：
- 对于一个待预测的样本，在训练集中找到与其最相似的K个样本
- 根据这K个近邻样本的标签，通过投票（分类）或平均（回归）的方式确定预测结果

**算法流程**：

1. **确定参数K**：选择近邻的数量K值
2. **计算距离**：计算待预测样本与训练集中所有样本之间的距离
3. **选择近邻**：找出距离最近的K个训练样本
4. **做出预测**：
   - **分类任务**：统计K个近邻中各类别的数量，选择数量最多的类别作为预测结果
   - **回归任务**：计算K个近邻标签的平均值（或加权平均值）作为预测结果

**距离度量方法**：

KNN算法中常用的距离度量方法包括：

- **欧氏距离（Euclidean Distance）**：
  $$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$
  适用于连续型特征，是最常用的距离度量方法

- **曼哈顿距离（Manhattan Distance）**：
  $$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$
  适用于网格状路径规划或高维稀疏数据

- **闵可夫斯基距离（Minkowski Distance）**：
  $$d(x, y) = \left(\sum_{i=1}^{n}|x_i - y_i|^p\right)^{\frac{1}{p}}$$
  当p=1时为曼哈顿距离，p=2时为欧氏距离

- **切比雪夫距离（Chebyshev Distance）**：
  $$d(x, y) = \max_{i}|x_i - y_i|$$
  适用于棋盘距离计算

**加权KNN**：

在标准KNN中，所有K个近邻对预测结果的贡献是相同的。加权KNN则根据距离给不同的近邻赋予不同的权重，通常距离越近的样本权重越大：

- **反距离加权**：权重与距离成反比
  $$w_i = \frac{1}{d(x, x_i)}$$

- **高斯加权**：使用高斯函数计算权重
  $$w_i = e^{-\frac{d(x, x_i)^2}{2\sigma^2}}$$

### 1.2 关键参数

KNN算法的性能受多个参数影响，合理选择这些参数对于获得良好的预测效果至关重要。

**K值的选择**：

K值是KNN算法中最重要的参数，直接影响模型的复杂度和性能：

- **K值较小**：
  - 模型复杂度高，对噪声敏感
  - 容易过拟合
  - 决策边界不规则

- **K值较大**：
  - 模型复杂度低，更加平滑
  - 容易欠拟合
  - 可能包含不相关的样本

**K值选择方法**：
- **交叉验证**：通过交叉验证选择使验证误差最小的K值
- **经验法则**：通常选择K = √n（n为训练样本数）
- **奇数原则**：分类问题中，K通常选择奇数，避免平票情况

**距离度量选择**：

- **欧氏距离**：适用于特征尺度相近的连续型数据
- **曼哈顿距离**：适用于高维数据或存在异常值的情况
- **标准化距离**：当特征尺度差异较大时，应先进行特征标准化

**权重设置**：

- **uniform（均匀权重）**：所有近邻权重相同
- **distance（距离权重）**：权重与距离成反比

**其他参数**：

- **algorithm（搜索算法）**：
  - **brute**：暴力搜索，适用于小数据集
  - **kd_tree**：KD树，适用于低维数据（维度<20）
  - **ball_tree**：球树，适用于高维数据
  - **auto**：自动选择最优算法

- **leaf_size（叶子大小）**：
  - 影响KD树或球树的构建和查询速度
  - 默认值通常为30

- **metric（距离度量）**：
  - 指定距离计算方法
  - 可选值包括'minkowski'、'euclidean'、'manhattan'等

- **p（闵可夫斯基距离参数）**：
  - p=1为曼哈顿距离
  - p=2为欧氏距离

### 1.3 优缺点

**优点**：

1. **简单直观**：算法思想简单，易于理解和实现
2. **无训练过程**：属于懒惰学习，无需训练模型，直接使用训练数据进行预测
3. **适应性强**：对数据分布没有假设，适用于各种数据分布
4. **可解释性好**：预测结果可以直观地解释为近邻样本的投票结果
5. **多用途**：既可以用于分类，也可以用于回归
6. **在线学习**：可以方便地添加新的训练样本，无需重新训练

**缺点**：

1. **预测速度慢**：需要计算待预测样本与所有训练样本的距离，时间复杂度高
2. **存储开销大**：需要存储所有训练数据
3. **维度灾难**：在高维空间中，距离度量失去意义，性能急剧下降
4. **对异常值敏感**：异常值会显著影响距离计算和预测结果
5. **对特征尺度敏感**：不同特征的量纲会影响距离计算，需要进行特征标准化
6. **类别不平衡问题**：当某些类别的样本数量远多于其他类别时，预测结果会偏向多数类
7. **难以处理缺失值**：需要额外的处理方法来处理缺失值

**适用场景**：

- 数据集规模较小
- 特征维度较低
- 数据分布不规则
- 需要可解释性强的模型
- 实时性要求不高的场景

**不适用场景**：

- 大规模数据集
- 高维稀疏数据
- 实时性要求高的场景
- 数据分布极度不平衡

### 1.4 API使用

#### 1.4.1 KNN分类器

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[2, 1], [3, 1], [1, 4], [2, 6]])
y = np.array([0, 0, 1, 1])  # 分类标签

# 定义KNN分类模型
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X, y)

# 预测
x = np.array([4, 9])
print(knn.predict([[4, 9]]))  # 输出: [0]
x_class = knn.predict([[4, 9]])

# 可视化
fig, ax = plt.subplots()
ax.axis('equal')
# 使用布尔索引将点分开
X1 = X[y == 0]
X2 = X[y == 1]
colors = ["C0", "C1"]
plt.scatter(X1[:, 0], X1[:, 1], color=colors[0])
plt.scatter(X2[:, 0], X2[:, 1], color=colors[1])
# 画出新点的预测
x_color = colors[0] if x_class == 0 else colors[1]
plt.scatter(4, 9, color=x_color)
plt.show()
```

#### 1.4.2 KNN回归器

```python
from sklearn.neighbors import KNeighborsRegressor

# 准备数据
X = [[2, 1], [3, 1], [1, 4], [2, 6]]
y = [0.5, 0.33, 4, 3]  # 回归目标值

# KNN回归模型
knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn.fit(X, y)

# 预测
x = [[4, 9]]
x_pred = knn.predict(x)
print(x_pred)  # 输出: [3.38208553]
```

## 2. 常见距离度量方法（了解）

距离度量是KNN算法的核心，不同的距离度量方法适用于不同的数据类型和应用场景。本节详细介绍KNN算法中常用的几种距离度量方法。

### 2.1 欧氏距离

欧氏距离（Euclidean Distance）是最常用的距离度量方法，它表示两个点在n维空间中的直线距离。

**数学公式**：
$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**特点**：
- 直观易懂，计算简单
- 适用于连续型特征
- 对异常值敏感
- 要求特征尺度相近，否则需要进行标准化

**适用场景**：
- 特征维度较低且尺度相近的数据
- 数据分布近似高斯分布
- 没有明显异常值的数据集

### 2.2 曼哈顿距离

曼哈顿距离（Manhattan Distance），也称为城市街区距离或L1距离，表示两个点在标准坐标系上的绝对轴距之和。

**数学公式**：
$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

**特点**：
- 对异常值不敏感
- 适用于高维数据
- 计算复杂度低于欧氏距离
- 在网格状路径规划中更适用

**适用场景**：
- 高维稀疏数据
- 存在异常值的数据集
- 网格状路径规划问题
- 特征维度较高的情况

### 2.3 切比雪夫距离

切比雪夫距离（Chebyshev Distance），也称为棋盘距离，表示两个点在各坐标数值差的最大值。

**数学公式**：
$$d(x, y) = \max_{i}|x_i - y_i|$$

**特点**：
- 只关注最大差异维度
- 适用于各维度重要性相同的情况
- 计算简单，只需找出最大差值

**适用场景**：
- 棋盘距离计算
- 各维度重要性相同的场景
- 需要关注最大偏差的应用

### 2.4 闵可夫斯基距离

闵可夫斯基距离（Minkowski Distance）是欧氏距离和曼哈顿距离的推广形式，通过参数p可以灵活地在这两种距离之间切换。

**数学公式**：
$$d(x, y) = \left(\sum_{i=1}^{n}|x_i - y_i|^p\right)^{\frac{1}{p}}$$

**参数说明**：
- 当p=1时，闵可夫斯基距离等于曼哈顿距离
- 当p=2时，闵可夫斯基距离等于欧氏距离
- 当p→∞时，闵可夫斯基距离趋近于切比雪夫距离

**特点**：
- 通用性强，通过调整p值可以适应不同场景
- p值越大，对较大差异的维度越敏感
- 提供了距离度量方法的统一框架

**适用场景**：
- 需要根据数据特点灵活选择距离度量方法的场景
- 作为其他距离度量方法的通用形式

## 3. 归一化与标准化

在KNN算法中，由于需要计算样本之间的距离，特征的尺度对距离计算结果有重要影响。当不同特征的取值范围差异较大时，取值范围大的特征会在距离计算中占据主导地位，导致其他特征的影响被掩盖。因此，在使用KNN算法之前，通常需要对特征进行归一化或标准化处理。

### 3.1 归一化

归一化（Normalization）是将特征缩放到一个固定的范围，通常是[0, 1]或[-1, 1]。

**最小-最大归一化（Min-Max Normalization）**：

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**代码实现**：

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 准备数据
X = [[2, 1], [3, 1], [1, 4], [2, 6]]

# 归一化，区间设置为(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = scaler.fit_transform(X)
print(X_normalized)
# 输出:
# [[ 0.  -1. ]
#  [ 1.  -1. ]
#  [-1.   0.2]
#  [ 0.   1. ]]
```

**特点**：
- 将数据缩放到固定范围[0, 1]
- 保留了原始数据的分布形状
- 对异常值敏感
- 适用于数据分布不明确的情况

**适用场景**：
- 需要将数据映射到特定范围
- 数据没有明显的异常值
- 对数据的分布形状没有要求

**注意事项**：
- 测试集需要使用训练集的min和max进行归一化
- 新数据可能超出[0, 1]范围，需要特殊处理

### 3.2 标准化

标准化（Standardization），也称为Z-score标准化，是将特征转换为均值为0、标准差为1的标准正态分布。

**Z-score标准化**：

$$x_{std} = \frac{x - \mu}{\sigma}$$

其中，$\mu$是特征的均值，$\sigma$是特征的标准差。

**代码实现**：

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# 准备数据
X = [[2, 1], [3, 1], [1, 4], [2, 6]]
X = np.array(X)

# 方法1: 使用sklearn的StandardScaler
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
print(X_standardized)
# 输出:
# [[ 0.         -0.94280904]
#  [ 1.41421356 -0.94280904]
#  [-1.41421356  0.47140452]
#  [ 0.          1.41421356]]

# 方法2: 手动计算
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_manual = (X - mean) / std
print(X_manual)
```

**特点**：
- 将数据转换为标准正态分布
- 对异常值不敏感
- 保留了数据的相对分布关系
- 适用于数据近似正态分布的情况

**适用场景**：
- 数据存在异常值
- 数据近似正态分布
- 需要使用基于距离的算法（如KNN、SVM等）
- 数据特征尺度差异较大

**注意事项**：
- 测试集需要使用训练集的均值和标准差进行标准化
- 标准化后的数据范围不限于[0, 1]

**归一化与标准化的选择**：

| 特点 | 归一化 | 标准化 |
|------|--------|--------|
| 数据范围 | [0, 1] | 无限制 |
| 对异常值敏感性 | 高 | 低 |
| 数据分布 | 保持原分布 | 转换为标准正态分布 |
| 适用算法 | 神经网络、KNN | 大多数机器学习算法 |

## 4. 案例：心脏病预测

本节通过一个完整的心脏病预测案例，展示KNN算法在实际问题中的应用流程，包括数据加载、预处理、模型训练、评估和保存等步骤。

### 4.1 数据集说明

心脏病预测是一个典型的二分类问题，目标是根据患者的各项生理指标预测其是否患有心脏病。

**数据集特征**：
- **年龄（Age）**：患者的年龄
- **性别（Sex）**：患者的性别（1=男性，0=女性）
- **胸痛类型（Chest Pain Type）**：胸痛的类型（0-3）
- **静息血压（Resting Blood Pressure）**：静息状态下的血压（mm Hg）
- **血清胆固醇（Serum Cholesterol）**：血清胆固醇水平（mg/dl）
- **空腹血糖（Fasting Blood Sugar）**：空腹血糖水平（>120 mg/dl为1，否则为0）
- **静息心电图（Resting ECG）**：静息心电图结果（0-2）
- **最大心率（Max Heart Rate）**：达到的最大心率
- **运动诱发心绞痛（Exercise Induced Angina）**：运动是否诱发心绞痛（1=是，0=否）
- **ST段压低（ST Depression）**：运动相对于休息的ST段压低程度
- **ST段斜率（ST Slope）**：峰值运动ST段的斜率（0-2）
- **主要血管数（Major Vessels）**：荧光透视着色的主要血管数（0-3）
- **地中海贫血（Thalassemia）**：地中海贫血类型（0-3）

**目标变量**：
- **Target**：是否患有心脏病（1=是，0=否）

**数据集特点**：
- 包含数值型和分类型特征
- 特征尺度差异较大，需要进行标准化处理
- 属于医学诊断领域，对模型准确性要求较高

### 4.2 加载数据集

```python
import pandas as pd

# 加载数据集
data = pd.read_csv('../data/heart.csv')

# 处理缺失值
data = data.dropna()

# 查看数据信息
data.info()
print(data.head())
```

### 4.3 数据集划分

```python
from sklearn.model_selection import train_test_split

# 划分特征和标签
X = data.drop('目标', axis=1)
y = data['目标']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

### 4.4 特征工程

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 类别型特征
categorical_features = ["胸痛类型", "静息心电图", "ST段斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动诱发心绞痛"]

# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        # 对数值型特征进行标准化
        ("num", StandardScaler(), 
         ["年龄", "静息血压", "血清胆固醇", "最大心率", "ST段压低", "主要血管数"]),
        # 对类别型特征进行独热编码，使用 drop="first" 避免多重共线性
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        # 二元特征不进行处理
        ("binary", "passthrough", binary_features),
    ]
)

# 执行特征转换
x_train = preprocessor.fit_transform(X_train)  # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(X_test)  # 使用训练集计算的信息对测试集进行转换

print(f"转换后的训练集形状: {x_train.shape}")
print(f"转换后的测试集形状: {x_test.shape}")
```

### 4.5 模型训练与评估

```python
from sklearn.neighbors import KNeighborsClassifier

# 创建模型
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(x_train, y_train)

# 模型评估，计算预测准确率
accuracy = knn.score(x_test, y_test)
print(f"模型准确率: {accuracy:.5f}")
```

### 4.6 模型的保存

```python
import joblib

# 保存模型
joblib.dump(knn, 'heart_disease_model.pkl')

# 加载模型，对新数据进行预测
knn_load = joblib.load('heart_disease_model.pkl')
print(f"预测类别：{knn_load.predict(x_test[10:11])}, 真实类别：{y_test.iloc[10]}")
```

## 5. 模型评估与超参数调优

超参数调优是机器学习模型开发中的重要环节，通过系统地搜索最优的超参数组合，可以显著提升模型的性能。本节介绍网格搜索方法及其在KNN算法中的应用。

### 5.1 网格搜索

网格搜索（Grid Search）是一种系统性的超参数调优方法，它通过遍历所有可能的超参数组合，找到使模型性能最优的参数配置。

**基本原理**：

1. **定义参数网格**：指定每个超参数的可能取值范围
2. **遍历所有组合**：对每个超参数组合进行模型训练和评估
3. **选择最优组合**：根据评估指标选择性能最好的参数组合

**网格搜索的变体**：

- **普通网格搜索**：遍历所有可能的参数组合
- **随机网格搜索**：从参数空间中随机采样组合进行评估
- **分层网格搜索**：先进行粗粒度搜索，再在优选的子空间进行细粒度搜索

**评估策略**：

- **交叉验证**：使用k折交叉验证评估每个参数组合的性能
- **验证集划分**：将训练集划分为训练子集和验证集

**优缺点**：

**优点**：
- 系统性搜索，不会遗漏重要的参数组合
- 可以并行化执行，提高效率
- 结果可解释性强

**缺点**：
- 计算成本高，特别是参数空间较大时
- 需要预先定义参数范围，可能错过最优值
- 对连续型参数不够高效

### 5.2 对心脏病预测模型进行超参数调优

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# 创建knn分类器
knn = KNeighborsClassifier()

# 定义网格搜索参数列表
param_grid = {
    'n_neighbors': list(range(1, 11)),
    'weights': ['uniform', 'distance'],
}

# 创建网格搜索对象，使用10折交叉验证
grid_search = GridSearchCV(knn, param_grid, cv=10)

# 模型训练
grid_search.fit(x_train, y_train)

# 打印模型评估结果
results = pd.DataFrame(grid_search.cv_results_).to_string()
print(results)

# 获取最佳模型和得分
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f"最佳模型: {best_model}")
print(f"最佳得分: {best_score:.5f}")

# 使用最佳模型进行预测
knn = grid_search.best_estimator_
print(f"测试集准确率: {knn.score(x_test, y_test):.5f}")
```

**代码说明**：

1. **定义参数网格**：
   - `n_neighbors`：从1到10的整数
   - `weights`：'uniform'（均匀权重）和'distance'（距离权重）

2. **创建GridSearchCV对象**：
   - `cv=10`：使用10折交叉验证
   - 评估所有参数组合（10×2=20种组合）

3. **训练与评估**：
   - 在训练集上进行网格搜索
   - 查看所有组合的详细结果
   - 获取最佳模型和对应的交叉验证得分

4. **最终评估**：
   - 使用最佳模型在测试集上进行最终评估
   - 比较训练集和测试集的表现

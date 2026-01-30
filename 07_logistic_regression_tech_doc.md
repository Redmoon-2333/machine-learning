# 逻辑回归技术文档

## 5. 逻辑回归

逻辑回归（Logistic Regression）是一种经典的分类算法，虽然名字中带有"回归"二字，但它实际上是用于解决分类问题的监督学习算法。逻辑回归通过sigmoid函数将线性回归的输出映射到(0,1)区间，表示样本属于某一类别的概率。

## 5.1 逻辑回归简介

### 5.1.1 什么是逻辑回归

逻辑回归是一种广义的线性模型，用于解决二分类问题。它的核心思想是：首先通过线性函数对输入特征进行组合，然后通过sigmoid函数将线性输出转换为概率值。

**基本形式**：

对于二分类问题，逻辑回归模型可以表示为：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

其中：
- $\mathbf{x}$：输入特征向量
- $\mathbf{w}$：权重向量
- $b$：偏置项
- $\sigma(\cdot)$：sigmoid函数

**sigmoid函数**：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

sigmoid函数的特点：
- 输出范围在(0,1)之间
- 当$z \to +\infty$时，$\sigma(z) \to 1$
- 当$z \to -\infty$时，$\sigma(z) \to 0$
- 当$z = 0$时，$\sigma(z) = 0.5$
- 函数图像呈S形，处处可导

**决策边界**：

逻辑回归的决策边界是线性的（对于线性逻辑回归）：

$$\mathbf{w}^T\mathbf{x} + b = 0$$

当$P(y=1|\mathbf{x}) \geq 0.5$时，预测为正类；否则预测为负类。

**与线性回归的区别**：

| 特性 | 线性回归 | 逻辑回归 |
|------|----------|----------|
| 任务类型 | 回归（预测连续值） | 分类（预测离散类别） |
| 输出范围 | $(-\infty, +\infty)$ | $(0, 1)$ |
| 输出含义 | 目标变量的预测值 | 属于正类的概率 |
| 损失函数 | 均方误差（MSE） | 对数损失（Log Loss） |
| 激活函数 | 无 | sigmoid函数 |

### 5.1.2 逻辑回归应用场景

逻辑回归广泛应用于各个领域，特别适合以下场景：

**医疗诊断**：
- 疾病风险评估（患病/未患病）
- 癌症检测（良性/恶性）
- 药物有效性预测

**金融领域**：
- 信用评分（违约/不违约）
- 欺诈检测（欺诈/正常）
- 贷款审批（通过/拒绝）

**营销领域**：
- 客户流失预测（流失/不流失）
- 广告点击率预测（点击/不点击）
- 购买意愿预测（购买/不购买）

**其他领域**：
- 邮件分类（垃圾邮件/正常邮件）
- 情感分析（正面/负面）
- 图像识别（是/否）

**适用条件**：
- 二分类问题
- 特征与目标之间存在近似线性关系（在log-odds尺度上）
- 需要概率输出的场景
- 对模型可解释性有要求

**不适用场景**：
- 多分类问题（需要使用扩展方法）
- 特征与目标之间存在复杂非线性关系
- 类别极度不平衡且没有处理

### 5.1.3 逻辑回归损失函数

逻辑回归使用对数损失（Log Loss）或交叉熵损失（Cross-Entropy Loss）作为损失函数。

**对数损失函数**：

对于单个样本，损失函数为：

$$L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

其中：
- $y \in \{0, 1\}$：真实标签
- $\hat{y} = P(y=1|\mathbf{x})$：预测概率

**直观理解**：

- 当$y=1$时，$L = -\log(\hat{y})$，希望$\hat{y}$接近1，损失接近0
- 当$y=0$时，$L = -\log(1-\hat{y})$，希望$\hat{y}$接近0，损失接近0

**整体损失函数**：

对于$n$个样本，平均损失为：

$$J(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**与极大似然估计的关系**：

对数损失函数实际上等价于负的对数似然函数。在假设样本独立同分布的情况下，最大化似然函数等价于最小化对数损失函数。

**似然函数**：

$$L(\mathbf{w}, b) = \prod_{i=1}^{n}\hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}$$

**对数似然函数**：

$$\ln L(\mathbf{w}, b) = \sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

因此，最小化对数损失函数等价于最大化对数似然函数。

### 5.1.4 损失函数的梯度（了解）

为了使用梯度下降法优化逻辑回归模型，需要计算损失函数对参数的梯度。

**梯度推导**：

对于单个样本，损失函数为：

$$L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

其中$\hat{y} = \sigma(z)$，$z = \mathbf{w}^T\mathbf{x} + b$。

首先，sigmoid函数的导数：

$$\frac{d\sigma(z)}{dz} = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$$

然后，计算损失函数对$z$的偏导：

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

$$= \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})$$

$$= -y(1-\hat{y}) + (1-y)\hat{y}$$

$$= \hat{y} - y$$

因此，损失函数对权重$w_j$的偏导为：

$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_j} = (\hat{y} - y)x_j$$

损失函数对偏置$b$的偏导为：

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = \hat{y} - y$$

**向量形式**：

对于$n$个样本，梯度为：

$$\nabla_{\mathbf{w}} J = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$$

$$\nabla_{b} J = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

**参数更新**：

使用梯度下降法更新参数：

$$\mathbf{w} := \mathbf{w} - \alpha \nabla_{\mathbf{w}} J$$

$$b := b - \alpha \nabla_{b} J$$

其中$\alpha$为学习率。

### 5.1.5 API使用

#### sklearn中的LogisticRegression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 创建示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression(
    solver='lbfgs',           # 优化算法
    penalty='l2',             # 正则化类型
    C=1.0,                    # 正则化强度的倒数
    class_weight=None,        # 类别权重
    max_iter=100,             # 最大迭代次数
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 评估
print(f"准确率: {model.score(X_test, y_test):.4f}")
print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_}")
```

#### 关键参数详解

**solver：优化算法**

| 算法 | 说明 | 适用场景 | 支持的正则化 |
|------|------|----------|-------------|
| `lbfgs` | 拟牛顿法（默认） | 中小规模数据集 | L2 |
| `newton-cg` | 牛顿法 | 中小规模数据集 | L2 |
| `liblinear` | 坐标下降法 | 小数据集 | L1, L2 |
| `sag` | 随机平均梯度下降 | 大规模数据集 | L2 |
| `saga` | 改进的随机梯度下降 | 大规模数据集 | L1, L2, ElasticNet |

**penalty：正则化类型**

- `l1`：L1正则化（Lasso），产生稀疏解，可用于特征选择
- `l2`：L2正则化（Ridge），默认选项，防止过拟合
- `elasticnet`：弹性网络，结合L1和L2正则化（需要saga求解器）
- `none`：无正则化（部分求解器支持）

**C：正则化强度**

- C是正则化强度的倒数，即 $C = \frac{1}{\lambda}$
- **C越小**，正则化强度越大，模型越简单，可能欠拟合
- **C越大**，正则化强度越小，模型越复杂，可能过拟合
- 通常通过交叉验证选择合适的C值

**class_weight：类别权重**

- `None`：所有类别权重相同（默认）
- `balanced`：自动根据类别频率调整权重，让模型更关注少数类
  - 计算公式：`n_samples / (n_classes * np.bincount(y))`
- 自定义字典：如 `{0: 1, 1: 3}` 表示类别0权重为1，类别1权重为3
- **适用场景**：处理类别不平衡问题

**其他重要参数**

- `max_iter`：最大迭代次数，默认100。如果收敛警告，可适当增大
- `tol`：收敛阈值，默认1e-4
- `multi_class`：多分类策略，`ovr`（一对多）或`multinomial`（Softmax）
- `random_state`：随机种子，保证结果可复现

#### 使用示例：处理类别不平衡

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 创建不平衡数据集
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)

# 使用class_weight='balanced'处理类别不平衡
model = LogisticRegression(
    solver='lbfgs',
    class_weight='balanced',  # 自动平衡类别权重
    random_state=42
)

model.fit(X, y)
```

#### 使用示例：L1正则化进行特征选择

```python
from sklearn.linear_model import LogisticRegression

# 使用liblinear求解器支持L1正则化
model = LogisticRegression(
    solver='liblinear',
    penalty='l1',      # L1正则化
    C=0.1,             # 较强的正则化
    random_state=42
)

model.fit(X_train, y_train)

# 查看哪些特征的系数被压缩为0（被筛选掉的特征）
print(f"非零系数数量: {(model.coef_ != 0).sum()}")
print(f"被筛选掉的特征: {(model.coef_ == 0).sum()}")
```

## 5.2 多分类任务

逻辑回归最初是为二分类问题设计的，但可以通过一些策略扩展到多分类问题。

### 5.2.1 一对多（OVR）

一对多（One-vs-Rest，OVR）策略是将多分类问题分解为多个二分类问题。

**基本思想**：

对于$K$个类别，训练$K$个二分类器。每个分类器将一个类别作为正类，其余所有类别作为负类。

**算法流程**：

1. 对于每个类别$k \in \{1, 2, ..., K\}$：
   - 将类别$k$的样本标记为正类（$y=1$）
   - 将其他所有类别的样本标记为负类（$y=0$）
   - 训练一个逻辑回归分类器，得到权重$\mathbf{w}_k$和偏置$b_k$

2. 预测时，将样本输入所有$K$个分类器，得到$K$个概率值：

$$P(y=k|\mathbf{x}) = \sigma(\mathbf{w}_k^T\mathbf{x} + b_k)$$

3. 选择概率最大的类别作为预测结果：

$$\hat{y} = \arg\max_{k} P(y=k|\mathbf{x})$$

**优点**：
- 简单直观，易于实现
- 可以并行训练多个分类器
- 适用于类别数量不太多的场景

**缺点**：
- 当类别数量很大时，需要训练很多分类器
- 每个分类器的训练数据不平衡（正类样本少，负类样本多）
- 概率输出可能不直接可比（因为没有归一化）

### 5.2.2 Softmax回归（多项逻辑回归）

Softmax回归（Softmax Regression）是逻辑回归在多分类问题上的自然扩展，也称为多项逻辑回归（Multinomial Logistic Regression）。

**基本思想**：

直接使用Softmax函数将线性输出转换为概率分布，所有类别的概率之和为1。

**Softmax函数**：

对于$K$个类别，Softmax函数定义为：

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x} + b_k}}{\sum_{j=1}^{K}e^{\mathbf{w}_j^T\mathbf{x} + b_j}}$$

其中：
- $\mathbf{w}_k$和$b_k$是第$k$类的权重和偏置
- 分子是第$k$类的得分（经过指数变换）
- 分母是所有类别得分的总和（归一化因子）

**特点**：

- 输出是一个概率分布，所有类别的概率之和为1
- 指数函数保证了概率值为正
- 当$K=2$时，Softmax回归等价于逻辑回归

**损失函数**：

Softmax回归使用交叉熵损失函数：

$$J(\mathbf{W}, \mathbf{b}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}y_{ik}\log(\hat{y}_{ik})$$

其中：
- $y_{ik}$是one-hot编码的真实标签（如果样本$i$属于类别$k$，则$y_{ik}=1$，否则为0）
- $\hat{y}_{ik} = P(y_i=k|\mathbf{x}_i)$是预测概率

**与OVR的比较**：

| 特性 | OVR | Softmax回归 |
|------|-----|-------------|
| 模型数量 | $K$个二分类器 | 1个多分类器 |
| 概率归一化 | 否 | 是 |
| 参数数量 | $K \times d$ | $K \times d$ |
| 训练效率 | 可并行 | 单次优化 |
| 适用场景 | 类别数较少 | 类别数较多 |

## 5.3 案例：手写数字识别

本节通过一个完整的手写数字识别案例，展示逻辑回归在实际问题中的应用流程。

### 5.3.1 数据集说明

手写数字识别是一个经典的多分类问题，目标是根据手写数字的图像识别出对应的数字（0-9）。

**常用数据集**：

**MNIST数据集**：
- 训练集：60,000张28×28像素的手写数字图像
- 测试集：10,000张28×28像素的手写数字图像
- 类别：10个数字（0-9）
- 图像格式：灰度图像，像素值范围0-255

**数据特点**：
- 图像已经过预处理，居中且大小归一化
- 每个样本有784个特征（28×28像素）
- 类别平衡，每个数字大约有6000个样本
- 是机器学习领域最常用的基准数据集之一

**业务背景**：

手写数字识别在邮政编码识别、银行支票处理、表单自动录入等场景有广泛应用。逻辑回归作为一个简单高效的基线模型，可以很好地解决这一问题。

### 5.3.2 逻辑回归实现手写数字识别

以下是使用逻辑回归实现手写数字识别的完整代码：

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# 1.加载数据集
data = pd.read_csv('../data/train.csv')

# 测试图像（可选，用于可视化）
# test_image = data.iloc[10,1:].values
# plt.imshow(test_image.reshape(28,28))
# plt.show()

# 2. 划分数据集
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程：归一化
# 将像素值从0-255归一化到0-1范围
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 定义模型和训练
# 由于数据量较大，需要增加max_iter以确保收敛
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 5. 模型评估
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# 6. 单样本测试
digits = X_test[123,:].reshape(1,-1)
print(f"预测结果: {model.predict(digits)}")
print(f"真实标签: {y_test.iloc[123]}")
print(f"预测概率: {model.predict_proba(digits)}")

# 7. 可视化预测结果
plt.imshow(digits.reshape(28,28))
plt.title(f"Predicted: {model.predict(digits)[0]}")
plt.show()
```

**代码说明**：

1. **数据加载**：从CSV文件加载MNIST数据集，每个样本有784个特征（28×28像素）和1个标签

2. **数据划分**：使用`train_test_split`将数据划分为训练集（80%）和测试集（20%）

3. **特征工程**：
   - 使用`MinMaxScaler`将像素值归一化到[0,1]范围
   - 归一化有助于加速模型收敛，提高训练效率

4. **模型训练**：
   - 使用默认的Softmax回归（`multi_class='auto'`会自动选择multinomial）
   - 设置`max_iter=10000`确保模型充分收敛

5. **模型评估**：使用`score`方法计算测试集上的准确率

6. **预测与可视化**：展示单个样本的预测结果和概率分布

## 5.4 案例：心脏病预测

本节通过心脏病预测案例，展示逻辑回归在二分类问题中的应用，以及如何处理混合类型的特征数据。

### 5.4.1 数据集说明

**数据集特征**：

心脏病数据集包含多种类型的特征：

| 特征类型 | 特征名称 |
|----------|----------|
| 数值型特征 | 年龄、静息血压、血清胆固醇、最大心率、ST段压低、主要血管数 |
| 类别型特征 | 胸痛类型、静息心电图、ST段斜率、地中海贫血 |
| 二元特征 | 性别、空腹血糖、运动诱发心绞痛 |

**目标变量**：
- 0：无心脏病
- 1：有心脏病

### 5.4.2 逻辑回归实现心脏病预测

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# 1. 加载数据集
data = pd.read_csv('../data/heart.csv')
data = data.dropna()  # 删除缺失值

# 2. 划分数据集
X = data.drop('目标', axis=1)
y = data['目标']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程
# 定义不同类型的特征
categorical_features = ["胸痛类型", "静息心电图", "ST段斜率", "地中海贫血"]  # 类别型特征
binary_features = ["性别", "空腹血糖", "运动诱发心绞痛"]  # 二元特征

# 创建列转换器，对不同类型的特征进行不同的处理
preprocessor = ColumnTransformer(
    transformers=[
        # 对数值型特征进行标准化
        ("num", StandardScaler(), ["年龄", "静息血压", "血清胆固醇", "最大心率", "ST段压低", "主要血管数"]),
        # 对类别型特征进行独热编码，使用 drop="first" 避免多重共线性
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        # 二元特征不进行处理
        ("binary", "passthrough", binary_features),
    ]
)

# 执行特征转换
x_train = preprocessor.fit_transform(X_train)  # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(X_test)  # 使用训练集计算的信息对测试集进行转换

# 4. 模型定义和训练
model = LogisticRegression()
model.fit(x_train, y_train)

# 5. 准确率计算，评估模型
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

**代码说明**：

1. **数据预处理**：
   - 删除含有缺失值的样本
   - 将特征和目标变量分离

2. **特征工程**：
   - **数值型特征**：使用`StandardScaler`进行标准化（均值为0，标准差为1）
   - **类别型特征**：使用`OneHotEncoder`进行独热编码，`drop="first"`避免多重共线性
   - **二元特征**：保持原样不变（`passthrough`）

3. **ColumnTransformer的作用**：
   - 允许对不同列应用不同的预处理方法
   - 统一管理所有特征转换流程
   - 确保训练集和测试集使用相同的转换参数

4. **模型训练与评估**：
   - 使用默认参数的逻辑回归模型
   - 通过`score`方法评估模型在测试集上的准确率

**关键点**：

- `fit_transform`只能用于训练集，它会计算并保存转换参数
- `transform`用于测试集，使用训练集计算的参数进行转换
- 这样可以避免数据泄露，确保模型评估的公正性

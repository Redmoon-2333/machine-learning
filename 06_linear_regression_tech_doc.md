# 线性回归技术文档

## 1. 线性回归简介

线性回归（Linear Regression）是机器学习中最基础、最经典的算法之一，用于建立自变量（特征）与因变量（目标）之间的线性关系模型。它既可以用于预测连续型数值，也可以作为理解更复杂算法的基础。

### 1.1 什么是线性回归

线性回归是一种监督学习算法，它假设目标变量与特征之间存在线性关系，通过拟合一条直线（或超平面）来预测目标值。

**基本形式**：

对于一元线性回归：
$$y = wx + b$$

对于多元线性回归：
$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \mathbf{w}^T\mathbf{x} + b$$

其中：
- $y$：目标变量（因变量）
- $x$：特征变量（自变量）
- $w$：权重（斜率）
- $b$：偏置（截距）

**核心思想**：
线性回归的核心是找到一组最优的权重参数$\mathbf{w}$和偏置$b$，使得模型预测值与真实值之间的误差最小。

**几何解释**：
- 在一维情况下，线性回归拟合的是一条直线
- 在二维情况下，线性回归拟合的是一个平面
- 在高维情况下，线性回归拟合的是一个超平面

**模型特点**：
- **简单可解释**：模型参数具有明确的物理意义
- **计算高效**：有解析解，计算速度快
- **基础性强**：是理解其他复杂算法的基础
- **适用性广**：可作为基准模型（Baseline）

### 1.2 线性回归应用场景

线性回归广泛应用于各个领域，特别适合以下场景：

**经济金融领域**：
- 股票价格预测
- 房价评估
- 销售额预测
- 风险评估

**医疗健康领域**：
- 疾病风险预测
- 药物剂量计算
- 生命体征预测

**工程领域**：
- 材料强度预测
- 能耗预测
- 质量检测

**市场营销领域**：
- 广告效果预测
- 客户价值评估
- 市场趋势分析

**适用条件**：
- 目标变量为连续型数值
- 特征与目标之间存在近似线性关系
- 数据量适中，特征维度不高
- 对模型可解释性有要求

**不适用场景**：
- 目标变量为离散类别（应使用分类算法）
- 特征与目标之间存在复杂非线性关系
- 数据存在严重的多重共线性
- 异常值较多且影响较大

### 1.3 API使用

#### 使用 sklearn 实现线性回归

**方法1：使用 LinearRegression（正规方程法）**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 准备数据
# 自变量，每周学习时长
X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])
# 因变量，数学考试成绩
y = np.array([55, 65, 70, 75, 85, 50, 60, 72, 80, 58])

# 2. 实例化线性回归模型
model = LinearRegression()

# 3. 模型训练
model.fit(X, y)

# 4. 打印模型参数
print("斜率 (coef_):", model.coef_)      # 输出: [2.904...]
print("截距 (intercept_):", model.intercept_)  # 输出: 41.103...

# 5. 预测新数据
x_new = np.array([[11]])
y_pred = model.predict(x_new)
print(f"学习11小时的预测成绩: {y_pred[0]:.2f}")

# 6. 可视化
x_line = np.arange(0, 15, 0.1).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, y, color='green', label='训练数据')
plt.plot(x_line, y_line, color='red', label='回归线')
plt.scatter(x_new, y_pred, color='blue', label='预测点')
plt.xlabel('学习时长（小时）')
plt.ylabel('考试成绩')
plt.legend()
plt.show()
```

**方法2：使用 SGDRegressor（随机梯度下降法）**

```python
from sklearn.linear_model import SGDRegressor
import numpy as np

# 1. 定义数据
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]

# 2. 定义模型
model = SGDRegressor(
    penalty=None,              # 无正则化
    loss='squared_error',      # 损失函数：平方误差
    max_iter=1000000,          # 最大迭代次数
    eta0=1e-5,                 # 初始学习率
    learning_rate='constant',  # 常数学习率
    tol=1e-8                   # 收敛阈值
)

# 3. 训练模型
model.fit(X, y)

# 4. 打印模型参数
print("斜率 (coef_):", model.coef_)          # 输出: [2.904...]
print("截距 (intercept_):", model.intercept_)  # 输出: [41.103...]

# 5. 预测
x_new = [[11]]
y_pred = model.predict(x_new)
print(f"学习11小时的预测成绩: {y_pred[0]:.2f}")
```

**两种方法的比较**：

| 特性 | LinearRegression | SGDRegressor |
|------|------------------|--------------|
| 求解方法 | 正规方程法（解析解） | 随机梯度下降法（迭代求解） |
| 适用数据规模 | 中小规模（特征数 < 10000） | 大规模数据 |
| 计算速度 | 快（直接计算） | 较慢（需要迭代） |
| 超参数 | 无需调参 | 需要设置学习率等参数 |
| 在线学习 | 不支持 | 支持（partial_fit） |

## 2. 线性回归求解

线性回归的求解目标是找到最优的参数$\mathbf{w}$和$b$，使得预测值与真实值之间的误差最小。常用的求解方法包括解析法和迭代法。

### 2.1 损失函数

损失函数（Loss Function）用于衡量模型预测值与真实值之间的差异，线性回归通常使用均方误差（Mean Squared Error, MSE）作为损失函数。

**均方误差（MSE）**：

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - (\mathbf{w}^T\mathbf{x}_i + b))^2$$

其中：
- $n$：样本数量
- $y_i$：第$i$个样本的真实值
- $\hat{y}_i$：第$i$个样本的预测值

**损失函数的几何意义**：
- 表示预测值与真实值之间距离的平方和的平均值
- 平方项保证误差为正，且对大误差给予更大惩罚
- 目标是最小化损失函数，找到最优参数

**其他损失函数**：

- **平均绝对误差（MAE）**：
  $$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
  对异常值不敏感，但不可导

- **均方根误差（RMSE）**：
  $$RMSE = \sqrt{MSE}$$
  与目标变量同量纲，便于解释

- **R²分数（决定系数）**：
  $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$
  衡量模型解释数据变异的能力

**最小二乘法与极大似然估计的关系**：

最小二乘法（Least Squares）与极大似然估计（Maximum Likelihood Estimation, MLE）在特定条件下是等价的。

**假设条件**：
- 误差项 $\epsilon_i = y_i - \hat{y}_i$ 服从独立同分布的正态分布 $N(0, \sigma^2)$
- 即 $\epsilon_i \sim N(0, \sigma^2)$，且各 $\epsilon_i$ 相互独立

**极大似然估计推导**：

在上述假设下，观测值 $y_i$ 服从正态分布 $N(\mathbf{w}^T\mathbf{x}_i, \sigma^2)$，其概率密度函数为：

$$p(y_i|\mathbf{x}_i; \mathbf{w}, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)$$

对于 $n$ 个独立样本，似然函数为：

$$L(\mathbf{w}, \sigma) = \prod_{i=1}^{n} p(y_i|\mathbf{x}_i; \mathbf{w}, \sigma)$$

取对数得到对数似然函数：

$$\ln L(\mathbf{w}, \sigma) = \sum_{i=1}^{n} \ln p(y_i|\mathbf{x}_i; \mathbf{w}, \sigma)$$

$$= \sum_{i=1}^{n} \ln \left[ \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right) \right]$$

$$= \sum_{i=1}^{n} \left[ \ln\left(\frac{1}{\sqrt{2\pi}\sigma}\right) - \frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2} \right]$$

$$= -\frac{n}{2}\ln(2\pi) - n\ln\sigma - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

**关键结论**：

为使似然函数最大，需求解 $\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 = \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f(\mathbf{x}_i))^2$ 的最小值。

发现其与均方误差直接相关。即**最大化对数似然函数等价于最小化均方误差**：

$$\min_{\mathbf{w}} \sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

这正是最小二乘法的损失函数！

**总结**：
- 当误差服从独立同分布的正态分布时，**最小二乘法等价于极大似然估计**
- 这解释了为什么最小二乘法在统计学中如此重要
- 正态分布假设使得最小二乘法具有良好的统计性质（如无偏性、有效性等）

### 2.2 一元线性回归解析解

对于一元线性回归$y = wx + b$，可以通过最小二乘法直接求解最优参数。

**最小二乘法原理**：

最小化损失函数：
$$J(w, b) = \sum_{i=1}^{n}(y_i - (wx_i + b))^2$$

**求解过程**：

1. 对$w$和$b$分别求偏导并令其为0：

$$\frac{\partial J}{\partial w} = -2\sum_{i=1}^{n}x_i(y_i - wx_i - b) = 0$$

$$\frac{\partial J}{\partial b} = -2\sum_{i=1}^{n}(y_i - wx_i - b) = 0$$

2. 解方程组得到：

$$w = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$b = \bar{y} - w\bar{x}$$

其中$\bar{x}$和$\bar{y}$分别是$x$和$y$的均值。

**几何意义**：
- 最优直线经过样本点的中心$(\bar{x}, \bar{y})$
- 斜率$w$表示$x$每增加一个单位，$y$的平均变化量
- 截距$b$表示当$x=0$时，$y$的预测值

### 2.3 正规方程法（解析法）

正规方程法（Normal Equation）是一种用于求解线性回归的解析解的方法。它基于最小二乘法，通过求解矩阵方程来直接获得参数值。

**损失函数的矩阵形式**：

将损失函数转换为矩阵形式：

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(f(\mathbf{x}_i) - y_i)^2$$

$$= \frac{1}{n}\sum_{i=1}^{n}((\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_m x_{im}) - y_i)^2$$

$$= \frac{1}{n}\sum_{i=1}^{n}(\boldsymbol{\beta}^T \mathbf{x}_i - y_i)^2$$

$$= \frac{1}{n}\|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|_2^2$$

$$= \frac{1}{n}(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

**矩阵定义**：

- $\mathbf{X}$：$n \times (m+1)$ 的设计矩阵，包含一个全1的列（对应截距项）

$$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1m} \\ 1 & x_{21} & x_{22} & \cdots & x_{2m} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{nm} \end{bmatrix}$$

- $\boldsymbol{\beta}$：$(m+1) \times 1$ 的参数向量（包含截距项$\beta_0$）

$$\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_m \end{bmatrix}$$

- $\mathbf{y}$：$n \times 1$ 的因变量向量

$$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

**正规方程推导**：

对$\boldsymbol{\beta}$求偏导：

$$\frac{\partial MSE}{\partial \boldsymbol{\beta}} = \frac{\partial \left(\frac{1}{n}(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})\right)}{\partial \boldsymbol{\beta}}$$

$$= \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

当$\mathbf{X}^T\mathbf{X}$为满秩矩阵或正定矩阵时，令偏导等于0：

$$\frac{2}{n}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} - \frac{2}{n}\mathbf{X}^T\mathbf{y} = 0$$

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

解得：

$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**优点**：
- 直接求解，无需迭代
- 不需要选择学习率
- 对于小规模数据计算速度快

**缺点**：
- 需要计算矩阵的逆，时间复杂度为$O(d^3)$
- 当特征维度很高时，计算代价大
- 当$\mathbf{X}^T\mathbf{X}$不可逆时（多重共线性），无法使用

**适用场景**：
- 特征维度较小（通常$d < 10000$）
- 数据量适中
- 没有严重的多重共线性问题

**重要说明**：
正规方程法适用于特征数量较少的情况。当特征数量较大时，计算逆矩阵的复杂度会显著增加（时间复杂度为$O(d^3)$），此时梯度下降法更为适用。

### 2.4 梯度下降法

当特征维度很高或数据量很大时，正规方程法计算代价过高，此时可以使用梯度下降法（Gradient Descent）迭代求解最优参数。

**基本原理**：

梯度下降法通过迭代更新参数，沿着损失函数梯度下降的方向逐步逼近最优解。

**参数更新公式**：

$$\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

其中$\alpha$为学习率（Learning Rate），控制每次更新的步长。

**梯度计算**：

对于线性回归，损失函数对参数的梯度为：

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

**梯度下降的变体**：

1. **批量梯度下降（Batch Gradient Descent）**：
   - 每次迭代使用全部训练数据
   - 梯度估计准确，收敛稳定
   - 计算量大，不适合大规模数据

2. **随机梯度下降（Stochastic Gradient Descent, SGD）**：
   - 每次迭代随机使用一个样本
   - 计算速度快，适合大规模数据
   - 梯度估计有噪声，收敛不稳定

3. **小批量梯度下降（Mini-batch Gradient Descent）**：
   - 每次迭代使用一小批样本（如32、64、128个）
   - 平衡了计算效率和收敛稳定性
   - 最常用的梯度下降方法

**学习率选择**：

- 学习率过大：可能导致震荡或发散
- 学习率过小：收敛速度太慢
- 常用策略：学习率衰减（Learning Rate Decay）

**特征缩放**：

使用梯度下降前，通常需要对特征进行标准化或归一化，使不同特征的尺度相近，加速收敛。

**优点**：
- 适用于大规模数据和高维特征
- 可以在线学习（逐步添加数据）
- 可以扩展到其他损失函数和正则化

**缺点**：
- 需要选择学习率等超参数
- 需要多次迭代，收敛速度依赖于数据
- 可能收敛到局部最优（线性回归中不存在此问题）

**梯度下降法求解示例**：

以学生学习时间与数学考试成绩的关系为例，演示梯度下降法的求解过程。

**数据准备**：

自变量（学习时长）：$\mathbf{x} = [5, 8, 10, 12, 15, 3, 7, 9, 14, 6]^T$

因变量（数学考试成绩）：$\mathbf{y} = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]^T$

**问题定义**：

求 $\boldsymbol{\beta} = [\beta_0, \beta_1]^T$ 使得损失函数 $J(\boldsymbol{\beta}) = \frac{1}{n}\sum_{i=1}^{n}(f(x_i) - y_i)^2$ 最小。

**矩阵转换**：

为计算方便，给 $\mathbf{x}$ 添加一列1，转换为设计矩阵：

$$\mathbf{X} = \begin{bmatrix} 1 & 5 \\ 1 & 8 \\ 1 & 10 \\ 1 & 12 \\ 1 & 15 \\ 1 & 3 \\ 1 & 7 \\ 1 & 9 \\ 1 & 14 \\ 1 & 6 \end{bmatrix}$$

则损失函数为：

$$J(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|_2^2$$

**梯度计算**：

损失函数的梯度为：

$$\nabla J(\boldsymbol{\beta}) = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

**梯度下降求解步骤**：

1. **初始化参数**：
   - $\boldsymbol{\beta}$ 初始值取 $[1, 1]^T$
   - 学习率 $\alpha$ 取 $0.01$

2. **迭代更新**：
   - 计算梯度：$\nabla J(\boldsymbol{\beta}) = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$
   - 更新参数：$\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \cdot \nabla J(\boldsymbol{\beta})$
   - 重复直到收敛

3. **收敛条件**：
   - 梯度范数小于某个阈值（如 $10^{-6}$）
   - 或损失函数变化小于某个阈值
   - 或达到最大迭代次数

**迭代过程示例**：

| 迭代次数 | 参数 $\boldsymbol{\beta}$ | 损失函数 $J(\boldsymbol{\beta})$ | 梯度 $\nabla J(\boldsymbol{\beta})$ | 更新后参数 |
|---------|---------------------------|----------------------------------|-------------------------------------|-----------|
| 初始 | $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ | 3311.3 | $\begin{bmatrix} -114.2 \\ -1067.6 \end{bmatrix}$ | - |
| 第1次 | $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ | 3311.3 | $\begin{bmatrix} -114.2 \\ -1067.6 \end{bmatrix}$ | $\begin{bmatrix} 2.142 \\ 11.676 \end{bmatrix}$ |
| 第2次 | $\begin{bmatrix} 2.142 \\ 11.676 \end{bmatrix}$ | 2589.9686 | $\begin{bmatrix} 78.1168 \\ 936.3284 \end{bmatrix}$ | $\begin{bmatrix} 1.3608 \\ 2.3127 \end{bmatrix}$ |
| ... | ... | ... | ... | ... |
| 第n次 | $\begin{bmatrix} 41.4507 \\ 2.8707 \end{bmatrix}$ | 2.9812 | $\begin{bmatrix} -2.34 \times 10^{-12} \\ 1.18 \times 10^{-13} \end{bmatrix}$ | 收敛 |

**参数更新计算（第1次迭代）**：

$$\boldsymbol{\beta}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad \nabla J(\boldsymbol{\beta}_1) = \begin{bmatrix} -114.2 \\ -1067.6 \end{bmatrix}$$

$$\boldsymbol{\beta}_2 = \boldsymbol{\beta}_1 - \alpha \cdot \nabla J(\boldsymbol{\beta}_1) = \begin{bmatrix} 1 \\ 1 \end{bmatrix} - 0.01 \times \begin{bmatrix} -114.2 \\ -1067.6 \end{bmatrix} = \begin{bmatrix} 2.142 \\ 11.676 \end{bmatrix}$$

**参数更新计算（第2次迭代）**：

$$\boldsymbol{\beta}_2 = \begin{bmatrix} 2.142 \\ 11.676 \end{bmatrix}, \quad \nabla J(\boldsymbol{\beta}_2) = \begin{bmatrix} 78.1168 \\ 936.3284 \end{bmatrix}$$

$$\boldsymbol{\beta}_3 = \boldsymbol{\beta}_2 - \alpha \cdot \nabla J(\boldsymbol{\beta}_2) = \begin{bmatrix} 2.142 \\ 11.676 \end{bmatrix} - 0.01 \times \begin{bmatrix} 78.1168 \\ 936.3284 \end{bmatrix} = \begin{bmatrix} 1.3608 \\ 2.3127 \end{bmatrix}$$

**最终结果**：

经过n次迭代后，梯度接近于零，算法收敛：

$$\boldsymbol{\beta}_n = \begin{bmatrix} 41.4507 \\ 2.8707 \end{bmatrix}, \quad J(\boldsymbol{\beta}_n) = 2.9812, \quad \nabla J(\boldsymbol{\beta}_n) \approx \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

**结果解释**：

通过梯度下降法迭代求解，最终得到：
- $\beta_0 = 41.45$：截距，表示学习时间为0时的基础成绩约为41.45分
- $\beta_1 = 2.87$：斜率，表示每增加1小时学习时间，成绩平均提高约2.87分

**Python实现代码**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义损失函数
def J(beta, X, y, n):
    return np.sum((X @ beta - y)**2) / n

# 定义计算梯度的函数
def gradient(beta, X, y, n):
    return 2 * X.T @ (X @ beta - y) / n

# 1. 定义数据
X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])  # 自变量
y = np.array([[55], [65], [70], [75], [85], [50], [60], [72], [80], [58]])  # 因变量

n = X.shape[0]
# 2. 数据处理，X增加一列（偏置项）
X = np.c_[np.ones((n, 1)), X]

# 3. 初始化参数以及超参数
alpha = 0.01      # 学习率
max_iter = 10000  # 最大迭代次数
beta = np.array([[1], [1]])  # 初始参数

# 定义列表，保存参数变化轨迹
beta0_history = []
beta1_history = []

# 4. 重复迭代
while (np.abs(gradient_beta := gradient(beta, X, y, n)) > 1e-10).any() and (max_iter := max_iter - 1) >= 0:
    beta0_history.append(beta[0, 0])
    beta1_history.append(beta[1, 0])
    
    # 计算梯度
    gradient_beta = gradient(beta, X, y, n)
    # 更新参数
    beta = beta - alpha * gradient_beta
    
    # 每迭代10轮打印一次当前的参数值和损失值
    if max_iter % 10 == 0:
        print(f'beta: {beta.reshape(-1)}\tJ:{J(beta, X, y, n)}')

print(f"最终迭代次数: {10000 - max_iter}")
print(f"最终参数: beta0={beta[0,0]:.4f}, beta1={beta[1,0]:.4f}")

# 5. 可视化参数更新轨迹
plt.plot(beta0_history, beta1_history, 'b-', alpha=0.5)
plt.scatter(beta0_history[::100], beta1_history[::100], c='red', s=20)
plt.xlabel('beta0 (截距)')
plt.ylabel('beta1 (斜率)')
plt.title('梯度下降参数更新轨迹')
plt.grid(True)
plt.show()
```

**代码说明**：
- `J(beta, X, y, n)`：计算均方误差损失函数
- `gradient(beta, X, y, n)`：计算损失函数对参数的梯度
- `np.c_[np.ones((n, 1)), X]`：给X添加一列1，用于计算截距
- 使用while循环迭代，直到梯度小于阈值或达到最大迭代次数
- 保存参数历史，用于可视化梯度下降的收敛过程

## 3. 案例：广告投放效果预测

本节通过一个完整的广告投放效果预测案例，展示线性回归在实际问题中的应用流程。

### 3.1 数据集说明

广告投放效果预测是一个典型的回归问题，目标是根据广告投入预测销售额。

**数据集特征**：

- **TV**：电视广告投入（单位：千元）
- **Radio**：广播广告投入（单位：千元）
- **Newspaper**：报纸广告投入（单位：千元）

**目标变量**：

- **Sales**：销售额（单位：千元）

**数据集特点**：

- 包含3个数值型特征
- 目标变量为连续型数值
- 数据量适中，适合线性回归建模
- 可以分析不同广告渠道对销售额的贡献

**业务背景**：

企业希望通过分析历史广告投入与销售额的关系，优化广告预算分配，提高广告投放效果。线性回归模型可以帮助理解：
- 各广告渠道的投入产出比
- 预测给定预算下的销售额
- 为广告预算决策提供数据支持

### 3.2 使用线性回归预测广告投放效果

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# 1. 读取数据
data = pd.read_csv('../data/Advertising1.csv')

# 2. 数据预处理
data.dropna(inplace=True)  # 删除缺失值
data.drop(columns=data.columns[0], inplace=True)  # 删除第一列（索引列）

print("数据预览:")
print(data.head())

# 3. 划分训练集和测试集
X = data.drop('Sales', axis=1)  # 特征：TV, Radio, Newspaper
y = data['Sales']               # 目标：Sales

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 4. 特征工程：标准化
# 由于不同广告渠道的投入范围不同，需要进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 创建模型并训练

# 5.1 正规方程法（LinearRegression）
print("\n=== 正规方程法（LinearRegression）===")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

print("系数 (Coefficients):", model_lr.coef_)
print("截距 (Intercept):", model_lr.intercept_)

# 5.2 随机梯度下降法（SGDRegressor）
print("\n=== 随机梯度下降法（SGDRegressor）===")
model_sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model_sgd.fit(X_train_scaled, y_train)

print("系数 (Coefficients):", model_sgd.coef_)
print("截距 (Intercept):", model_sgd.intercept_)

# 6. 模型预测
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_sgd = model_sgd.predict(X_test_scaled)

# 7. 模型评估：使用均方误差（MSE）
print("\n=== 模型评估 ===")
print(f"LinearRegression MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"SGDRegressor MSE: {mean_squared_error(y_test, y_pred_sgd):.4f}")

# 8. 结果分析
print("\n=== 结果分析 ===")
feature_names = ['TV', 'Radio', 'Newspaper']
print("各广告渠道对销售额的影响程度（系数）:")
for i, name in enumerate(feature_names):
    print(f"  {name}: {model_lr.coef_[i]:.4f}")

print(f"\n模型解释:")
print(f"- TV广告每增加1个标准差，销售额平均增加 {model_lr.coef_[0]:.4f} 千元")
print(f"- Radio广告每增加1个标准差，销售额平均增加 {model_lr.coef_[1]:.4f} 千元")
print(f"- Newspaper广告每增加1个标准差，销售额平均增加 {model_lr.coef_[2]:.4f} 千元")
```

**代码说明**：

1. **数据读取与预处理**：
   - 使用 `pandas` 读取CSV数据
   - 删除缺失值和多余的索引列

2. **数据集划分**：
   - 使用 `train_test_split` 划分训练集（80%）和测试集（20%）
   - 设置 `random_state` 保证结果可复现

3. **特征标准化**：
   - 使用 `StandardScaler` 对特征进行标准化
   - 使不同量纲的特征具有相同的尺度

4. **模型训练**：
   - **LinearRegression**：使用正规方程法直接求解
   - **SGDRegressor**：使用随机梯度下降法迭代求解

5. **模型评估**：
   - 使用均方误差（MSE）评估模型性能
   - 比较两种方法的预测效果

6. **结果分析**：
   - 分析各广告渠道的系数，了解不同渠道对销售额的贡献
   - 系数越大，表示该渠道对销售额的影响越大

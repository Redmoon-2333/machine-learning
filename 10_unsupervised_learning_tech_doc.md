天# 无监督学习技术文档

## 1. 聚类

聚类（Clustering）是一种无监督学习方法，其目标是将数据集中的样本划分为若干个组（簇），使得同一组内的样本相似度较高，不同组间的样本相似度较低。

> **通俗理解**：想象你有一筐水果，里面有苹果、橙子、香蕉，但没有任何标签。你自然会按颜色、形状把它们分成三堆。这就是聚类——不需要告诉机器"这是苹果"，它自己发现相似的放一起。

### 1.1 聚类简介

聚类是无监督学习中最常见的任务之一，与分类不同，聚类不需要预先定义的类别标签。

**聚类的目标**：
- 发现数据中的内在结构
- 将相似的样本归为一类
- 最大化簇内相似度，最小化簇间相似度

**聚类的应用场景**：
- **客户分群**：根据消费行为将客户分为不同群体
- **图像分割**：将图像中相似的像素聚为一组
- **文档聚类**：将相似主题的文档归类
- **异常检测**：识别与大多数样本不同的异常点
- **数据压缩**：用簇中心代表一组数据

**聚类与分类的区别**：

| 特性 | 聚类 | 分类 |
|------|------|------|
| 学习类型 | 无监督学习 | 监督学习 |
| 标签需求 | 不需要标签 | 需要标签 |
| 目标 | 发现数据结构 | 预测类别 |
| 评估方式 | 内部指标为主 | 准确率等指标 |

> **一句话区分**：分类是"老师先告诉你答案，你学着判断"；聚类是"没有答案，你自己发现规律分组"。

### 1.2 常见聚类算法

**1. K-means（K均值聚类）**

K-means是最经典的聚类算法，通过迭代优化使每个样本到其所属簇中心的距离之和最小。

> **通俗理解**：想象你要在城市里开3家快递站，要让所有居民到最近的快递站距离之和最小。K-means就是不断调整快递站位置，直到找到最优解。

**算法流程**：
1. 随机选择$K$个样本作为初始簇中心
2. 将每个样本分配到距离最近的簇中心
3. 更新每个簇的中心为簇内所有样本的均值
4. 重复步骤2-3直到簇中心不再变化或达到最大迭代次数

> **形象类比**：就像班级分组——随机选三个组长，每个同学加入离自己最近的组长那组，然后每组重新选位置居中的人当组长，如此反复，直到分组稳定。

**优点**：
- 算法简单，易于实现
- 计算效率高，适合大规模数据
- 对球形簇效果好

**缺点**：
- 需要预先指定$K$值
- 对初始中心敏感
- 对异常值敏感
- 只能发现球形簇

> **缺点举例**：K-means擅长分“圆球状”的簇，但如果数据是“月牙形”或“环形”，它就会分错。就像用圆规画圆很容易，但画不出复杂形状。

**2. 层次聚类（Hierarchical Clustering）**

层次聚类通过构建树状结构来组织数据。

**两种策略**：
- **凝聚式（自底向上）**：每个样本初始为一个簇，逐步合并
- **分裂式（自顶向下）**：所有样本初始为一个簇，逐步分裂

**3. DBSCAN（基于密度的聚类）**

DBSCAN通过密度连接的方式发现任意形状的簇。

**核心概念**：
- **核心点**：在$\epsilon$邻域内有至少$MinPts$个样本
- **边界点**：在核心点的$\epsilon$邻域内，但自身不是核心点
- **噪声点**：既不是核心点也不是边界点

**优点**：
- 不需要预先指定簇数
- 可以发现任意形状的簇
- 能够识别噪声点

> **与K-means对比**：K-means像用“圆规”划分区域，DBSCAN像“顺藤摸瓜”——从一个点出发，只要附近有足够多的点就继续扩展，因此能发现任意形状的簇，也能自动把"落单"的点标记为噪声。

### 1.3 K-means API使用

**实战案例**：

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据
X, y = make_blobs(n_samples=300, centers=3, cluster_std=2)

# 画出散点图
fig, ax = plt.subplots(2, figsize=(8, 8))
ax[0].scatter(X[:, 0], X[:, 1], c=y, label="原始数据")
ax[0].set_title("原始数据")
ax[0].legend()

# 2. 定义模型并聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 3. 获取聚类结果
centers = kmeans.cluster_centers_

# 4. 得到所有样本点的分类标签
y_pred = kmeans.predict(X)
ax[1].scatter(X[:, 0], X[:, 1], c=y_pred, label="聚类结果")
ax[1].scatter(centers[:, 0], centers[:, 1], c='red', label="质心")
ax[1].set_title("K-means聚类结果")
ax[1].legend()
plt.show()
```

**基本用法**：

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# 2. 创建K-means模型
model = KMeans(
    n_clusters=4,          # 簇的数量
    init='k-means++',      # 初始化方法，k-means++更稳定
    n_init=10,             # 用不同初始值运行算法的次数
    max_iter=300,          # 最大迭代次数
    random_state=42
)

# 3. 训练模型
model.fit(X)

# 4. 获取聚类结果
labels = model.labels_           # 每个样本的簇标签
centers = model.cluster_centers_ # 簇中心
inertia = model.inertia_         # 簇内平方和

print(f"簇标签: {labels[:10]}")
print(f"簇中心:\n{centers}")
print(f"簇内平方和: {inertia:.2f}")

# 5. 预测新样本
new_samples = [[0, 0], [4, 4]]
predictions = model.predict(new_samples)
print(f"新样本预测: {predictions}")

# 6. 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='簇中心')
plt.title('K-means聚类结果')
plt.legend()
plt.show()
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `n_clusters` | 簇的数量$K$ |
| `init` | 初始化方法：`'k-means++'`（推荐）或`'random'` |
| `n_init` | 运行次数，选择最优结果 |
| `max_iter` | 单次运行的最大迭代次数 |
| `tol` | 收敛阈值 |

**如何选择K值——肘部法则**：

K-means算法需要预先指定簇的数量$K$，但在实际应用中往往不知道最佳的$K$值。肘部法则（Elbow Method）是一种常用的$K$值选择方法。

**肘部法则原理**：

1. **核心指标——簇内平方和（Inertia/SSE）**：
   $$SSE = \sum_{i=1}^{K}\sum_{x \in C_i}\|x - \mu_i\|^2$$
   表示所有样本到其所属簇中心的距离平方和，SSE越小，簇内样本越紧密。

2. **变化规律**：
   - 当$K$较小时，增加$K$会显著降低SSE（曲线下降陡峭）
   - 当$K$达到某个值后，继续增加$K$对SSE的降低作用变得很小（曲线趋于平缓）
   - 这个拐点就像人的胳膊弯曲处，的“肘部”

3. **选择依据**：
   肘部位置代表了“边际收益递减”的临界点——继续增加$K$带来的收益已经不值得为此付出的复杂度代价。

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 计算不同K值的簇内平方和
inertias = []
K_range = range(1, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)

# 绘制肘部曲线
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K值')
plt.ylabel('簇内平方和')
plt.title('肘部法则选择K值')
plt.show()
```

选择"肘部"位置（曲线开始变平缓的点）作为最优$K$值。

> **通俗解释**：肘部法就像省钱和体验的平衡。你可以开100家快递站让大家都住得近，但成本太高；开太少又不方便。曲线上“肘部”那个点，就是性价比最高的选择。

**肘部法则的局限性**：

| 问题 | 说明 |
|------|------|
| 肘部不明显 | 某些数据集的曲线平滑下降，没有明显拐点 |
| 主观判断 | “肘部”位置需要人工观察，不同人可能有不同结论 |
| 不适用复杂簇 | 对于大小不均、形状不规则的簇效果较差 |

**替代方法**：可以结合轮廓系数（Silhouette Score）等指标综合判断最佳$K$值。

### 1.4 聚类模型评估（了解）

聚类评估分为内部评估和外部评估两类。

**完整评估示例**：

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据
X, y = make_blobs(n_samples=300, centers=3, cluster_std=2)

# 2. 定义模型并聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 3. 获取聚类结果
centers = kmeans.cluster_centers_
y_pred = kmeans.predict(X)

# 4. 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, label="聚类结果")
plt.scatter(centers[:, 0], centers[:, 1], c='red', label="质心")
plt.legend()
plt.show()

# 5. 评价指标
print("簇内平方和:", kmeans.inertia_)
print("轮廓系数:", silhouette_score(X, y_pred))
print("Calinski-Harabasz指数:", calinski_harabasz_score(X, y_pred))
```

**内部评估指标**（不需要真实标签）：

**1. 轮廓系数（Silhouette Coefficient）**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

其中：
- $a(i)$：样本$i$到同簇其他样本的平均距离
- $b(i)$：样本$i$到最近其他簇所有样本的平均距离

轮廓系数范围为$[-1, 1]$，越接近1表示聚类效果越好。

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"轮廓系数: {score:.4f}")
```

**2. 簇内平方和（Inertia）**

$$\sum_{i=1}^{K}\sum_{x \in C_i}\|x - \mu_i\|^2$$

值越小表示簇内样本越紧密。

**3. CH指数（Calinski-Harabasz Index）**

CH指数也称为方差比准则（Variance Ratio Criterion），通过计算簇间方差与簇内方差的比值来评估聚类效果。

$$CH = \frac{SS_B / (K-1)}{SS_W / (n-K)}$$

其中：
- $SS_B$：簇间平方和（Between-cluster Sum of Squares），度量簇中心与总中心的分散程度
- $SS_W$：簇内平方和（Within-cluster Sum of Squares），度量样本与簇中心的紧密程度
- $K$：簇的数量
- $n$：样本总数

**CH指数的理解**：
- **值越大越好**：说明簇间分离度高（$SS_B$大）且簇内紧密度高（$SS_W$小）
- **无固定范围**：与轮廓系数不同，CH指数没有固定的取值范围
- **适合比较**：主要用于比较不同$K$值时的聚类效果

> **通俗理解**：CH指数就像评价班级分组的效果——好的分组应该是“组内团结”（组员之间很像）且“组间有区分”（不同组明显不同）。

```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, labels)
print(f"CH指数: {ch_score:.4f}")
```

**使用CH指数选择最佳K值**：

```python
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

ch_scores = []
K_range = range(2, 11)  # K从2开始，因为K=1时无法计算CH指数

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    ch_scores.append(calinski_harabasz_score(X, labels))

# 选择CH指数最大的K值
best_k = K_range[ch_scores.index(max(ch_scores))]
print(f"最佳K值: {best_k}")
```

**外部评估指标**（需要真实标签）：

**1. 调整兰德指数（Adjusted Rand Index, ARI）**

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels)
print(f"调整兰德指数: {ari:.4f}")
```

**2. 标准化互信息（Normalized Mutual Information, NMI）**

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, labels)
print(f"标准化互信息: {nmi:.4f}")
```

## 2. 降维（了解）

降维（Dimensionality Reduction）是将高维数据转换为低维表示的技术，在保留重要信息的同时减少特征维度。

> **通俗理解**：想象用100个指标描述一个学生（身高、体重、各科成绩、兴趣爱好...），但其实可能只需要"学习能力"和"运动能力"两个维度就能抓住主要特征。降维就是把100维压缩到更少维度，同时保留最重要的信息。

**降维的目的**：
- 减少计算复杂度
- 去除噪声和冗余特征
- 便于数据可视化
- 缓解维度灾难

### 2.1 奇异值分解

奇异值分解（Singular Value Decomposition, SVD）是一种矩阵分解技术，可以将任意矩阵分解为三个特殊矩阵的乘积。

**数学定义**：

对于$m \times n$矩阵$A$，SVD分解为：

$$A = U\Sigma V^T$$

其中：
- $U$：$m \times m$正交矩阵，列向量称为左奇异向量
- $\Sigma$：$m \times n$对角矩阵，对角元素为奇异值（非负，按降序排列）
- $V$：$n \times n$正交矩阵，列向量称为右奇异向量

**截断SVD**：

只保留前$k$个最大的奇异值，可以实现降维和数据压缩：

$$A_k = U_k\Sigma_k V_k^T$$

**代码实现**：

```python
import numpy as np

# 方法1：使用numpy的原生SVD
A = np.array([[1, 1], [2, 2], [0, 0]])
U, S, V = np.linalg.svd(A)
print("左奇异向量U:\n", U)
print("奇异值S:", S)
print("右奇异向量V:\n", V)

# 方法2：使用sklearn的随机化SVD（适合大规模数据）
from sklearn.utils.extmath import randomized_svd
U, S, V = randomized_svd(A, n_components=2)
print("\n随机化SVD结果:")
print("U:\n", U)
print("S:", S)
print("V:\n", V)
```

**应用场景**：
- 数据压缩
- 推荐系统（矩阵分解）
- 潜在语义分析（LSA）
- 图像压缩

> **形象类比**：SVD就像把一张大图片压缩成缩略图——保留主要特征，去掉细节，但还能认出是什么。只保留前几个最大的奇异值，就能实现信息的"去粗取精"。

### 2.2 主成分分析

主成分分析（Principal Component Analysis, PCA）是最常用的线性降维方法，通过正交变换将数据投影到方差最大的方向。

> **通俗理解**：想象用手电筒照一个3D雕像，在墙上投影出2D影子。PCA就是找到那个最佳角度，让影子能展现雕像最多的特征（而不是一个模糊的圆形）。第一主成分就是信息量最大的方向。

**核心思想**：
- 找到数据方差最大的方向（第一主成分）
- 找到与第一主成分正交且方差次大的方向（第二主成分）
- 依次类推，得到所有主成分

**数学原理**：

PCA本质上是对协方差矩阵进行特征值分解：

$$C = \frac{1}{n-1}X^TX$$

协方差矩阵$C$的特征向量即为主成分方向，特征值表示对应方向的方差大小。

**算法步骤**：
1. 对数据进行中心化（减去均值）
2. 计算协方差矩阵
3. 对协方差矩阵进行特征值分解
4. 选择前$k$个最大特征值对应的特征向量
5. 将数据投影到选定的特征向量上

**代码实现**：

```python
from sklearn.decomposition import PCA
import numpy as np

# 生成示例数据
X = np.random.randn(100, 3)  # 100个样本，3个特征

# 创建PCA模型，降到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"原始数据形状: {X.shape}")
print(f"降维后形状: {X_pca.shape}")
print(f"解释方差比: {pca.explained_variance_ratio_}")
```

**完整示例（包含可视化）**：

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 数据标准化（推荐）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 创建PCA模型
pca = PCA(n_components=2)  # 降到2维

# 3. 训练并转换数据
X_pca = pca.fit_transform(X_scaled)

# 4. 查看主成分信息
print(f"主成分方向:\n{pca.components_}")
print(f"解释方差: {pca.explained_variance_}")
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

# 5. 可视化降维结果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('PCA降维结果')
plt.show()
```

**如何选择主成分数量**：

```python
# 保留95%的方差
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"保留的主成分数: {pca.n_components_}")

# 绘制累计解释方差比曲线
pca_full = PCA()
pca_full.fit(X_scaled)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumsum)+1), cumsum, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%方差')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比')
plt.title('选择主成分数量')
plt.legend()
plt.show()
```

**PCA与SVD的关系**：
- PCA可以通过SVD实现
- 对中心化后的数据矩阵$X$进行SVD：$X = U\Sigma V^T$
- 主成分方向为$V$的列向量
- sklearn中的PCA默认使用SVD实现

**PCA的优缺点**：

| 优点 | 缺点 |
|------|------|
| 去除特征间的相关性 | 只能处理线性关系 |
| 降低模型复杂度 | 主成分的可解释性较差 |
| 便于可视化 | 对异常值敏感 |
| 去除噪声 | 需要标准化数据 |

> **简单比喻**：PCA就像拍集体照——100个人站成一团（3D），照片只有正面一张（2D）。摄影师要找到最佳角度，让照片尽可能看清每个人的脸。PCA就是自动找这个最佳角度。

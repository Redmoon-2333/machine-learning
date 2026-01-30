# 机器学习基础代码库

这是一个系统性的机器学习基础学习项目，涵盖了从特征工程到经典算法实现的完整流程。项目采用"理论+实践"的学习模式，通过详细的技术文档和可运行的代码示例，帮助学习者深入理解机器学习的核心概念和实现方法。

## 📚 项目特色

- **系统化学习路径**：从数据预处理到模型部署的完整流程
- **理论与实践结合**：每个概念都配有详细的数学推导和代码实现
- **丰富的案例实战**：涵盖分类、回归、聚类等经典机器学习任务
- **详细的技术文档**：9份完整的技术文档，深入讲解算法原理

## 📁 项目结构

```
.
├── ch02_base/                    # 第二章：机器学习基础
│   ├── feature/                  # 特征工程
│   │   ├── 1_variance_filter.ipynb      # 低方差过滤
│   │   ├── 2_pearson.ipynb              # 皮尔逊相关系数
│   │   ├── 3_spearman.ipynb             # 斯皮尔曼相关系数
│   │   └── 4_pca.ipynb                  # PCA降维
│   ├── metrics/                  # 模型评估指标
│   │   ├── 1_classification_test.ipynb  # 分类评估指标测试
│   │   └── 2_classification_report.py   # 分类评估报告生成
│   ├── 1_fitting_test.py         # 欠拟合/过拟合演示
│   ├── 2_regularization.py       # 正则化演示
│   ├── 3_gradient_descent1.ipynb # 梯度下降法（一）
│   └── 4_gradient_descent2.ipynb # 梯度下降法（二）
│
├── ch03_knn/                     # 第三章：KNN算法
│   ├── 1_classification_test.ipynb      # KNN分类测试
│   ├── 2_regression_test.ipynb          # KNN回归测试
│   ├── 3_scaler_test.ipynb              # 归一化与标准化测试
│   ├── 4_heart_disease.py               # 心脏病预测案例
│   └── heart_disease_model.pkl          # 保存的模型文件
│
├── ch04_linear_regression/       # 第四章：线性回归
│   ├── 1_lr_test.ipynb                  # 线性回归测试
│   ├── 2_gradient_descent.py            # 梯度下降法实现
│   ├── 3_sgd_test.ipynb                 # 随机梯度下降测试
│   └── 4_advertising.py                 # 广告投放效果预测
│
├── ch05_logistic_regression/     # 第五章：逻辑回归
│   ├── 1_api_test.ipynb                 # 逻辑回归API测试
│   ├── 2_heart_disease.py               # 心脏病预测（逻辑回归）
│   └── 3_digit_recognizer.py            # 手写数字识别
│
├── ch06_perceptron/              # 第六章：感知机
│   └── 1_logic_gate.py                  # 逻辑门实现
│
├── data/                         # 数据集目录
│   ├── advertising.csv           # 广告数据集
│   ├── Advertising1.csv          # 广告数据集（扩展版）
│   └── heart.csv                 # 心脏病数据集
│
├── 01_feature_engineering_tech_doc.md              # 特征工程技术文档
├── 02_model_solving_algorithms_tech_doc.md        # 模型求解算法技术文档
├── 03_model_evaluation_selection_tech_doc.md      # 模型评估和选择技术文档
├── 04_model_evaluation_tech_doc.md                # 模型评估技术文档
├── 05_knn_algorithm_tech_doc.md                   # KNN算法技术文档
├── 06_linear_regression_tech_doc.md               # 线性回归技术文档
├── 07_logistic_regression_tech_doc.md             # 逻辑回归技术文档
├── 08_perceptron_tech_doc.md                      # 感知机技术文档
└── 09_other_supervised_learning_tech_doc.md       # 其他监督学习算法技术文档
```

## 🎯 核心内容

### 1. 特征工程

特征工程是机器学习流程中的关键环节，直接影响模型性能。

**核心方法**：
- **低方差过滤法**：基于特征方差进行过滤，去除无信息特征
- **皮尔逊相关系数法**：基于线性相关关系选择特征
- **斯皮尔曼相关系数法**：基于单调相关关系选择特征
- **PCA降维**：通过线性变换将高维数据映射到低维空间

**应用场景**：
- 高维数据处理（基因数据、图像数据、文本数据）
- 特征冗余较多的场景（传感器数据、多源数据融合）
- 需要模型可解释性的场景（金融风控、医疗诊断）

📄 [查看特征工程技术文档](01_feature_engineering_tech_doc.md)

---

### 2. 模型求解算法

模型求解算法负责在给定模型结构和训练数据的情况下，找到最优的模型参数。

**核心算法**：
- **梯度下降法**：
  - 批量梯度下降（BGD）
  - 随机梯度下降（SGD）
  - 小批量梯度下降（Mini-batch GD）
- **进阶优化策略**：动量法、AdaGrad、RMSprop、Adam
- **解析解法**：通过数学推导直接求解最优参数
- **牛顿法和拟牛顿法**：BFGS、L-BFGS等二阶优化方法

**数学基础**：
- 一阶导数（梯度）与参数更新
- 学习率选择与收敛性分析
- 二阶导数（Hessian矩阵）与收敛加速

📄 [查看模型求解算法技术文档](02_model_solving_algorithms_tech_doc.md)

---

### 3. 模型评估和选择

模型评估和选择是机器学习流程中的关键环节，直接影响模型的泛化能力。

**核心概念**：
- **损失函数**：MSE、MAE、交叉熵损失等
- **经验误差与泛化误差**：模型在训练数据和未见数据上的表现差异
- **欠拟合与过拟合**：模型复杂度与数据拟合程度的关系
- **正则化**：L1正则化（Lasso）、L2正则化（Ridge）
- **交叉验证**：k折交叉验证、留一法等

**评估指标**：
- 分类任务：准确率、精确率、召回率、F1值、ROC-AUC
- 回归任务：MSE、MAE、RMSE、R²

📄 [查看模型评估和选择技术文档](03_model_evaluation_selection_tech_doc.md)

---

### 4. 模型评估实践

详细介绍模型评估的具体实现和应用。

**分类评估**：
- 混淆矩阵的生成与解读
- 分类报告（Classification Report）
- ROC曲线与AUC值的计算
- 多分类问题的评估策略

**回归评估**：
- 残差分析
- 预测值与真实值的对比可视化
- 模型性能的综合评估

📄 [查看模型评估技术文档](04_model_evaluation_tech_doc.md)

---

### 5. KNN算法

KNN（K-Nearest Neighbors）是一种经典的基于实例的学习算法。

**算法原理**：
- 通过计算样本之间的距离进行预测
- 既可以用于分类，也可以用于回归

**核心内容**：
- **距离度量**：欧氏距离、曼哈顿距离、闵可夫斯基距离、切比雪夫距离
- **特征预处理**：归一化与标准化处理
- **超参数调优**：K值选择、权重设置、网格搜索
- **实际案例**：心脏病预测完整流程

**应用场景**：
- 推荐系统
- 图像识别
- 文本分类
- 医学诊断

📄 [查看KNN算法技术文档](05_knn_algorithm_tech_doc.md)

---

### 6. 线性回归

线性回归是机器学习中最基础、最经典的算法之一。

**算法原理**：
- 通过拟合直线或超平面建立自变量与因变量的线性关系
- 既可以用于预测，也可以用于因果分析

**求解方法**：
- **正规方程法**：直接求解解析解，适用于中小规模数据
- **梯度下降法**：
  - 批量梯度下降
  - 随机梯度下降
  - 小批量梯度下降

**核心内容**：
- 损失函数（MSE）与极大似然估计的关系
- 特征标准化与归一化
- 模型评估与解释
- 实际案例：广告投放效果预测

��� [查看线性回归技术文档](06_linear_regression_tech_doc.md)

---

### 7. 逻辑回归

逻辑回归是一种经典的分类算法，虽然名字带有"回归"，但实际用于解决分类问题。

**算法原理**：
- 通过sigmoid函数将线性回归的输出映射到(0,1)区间
- 输出表示样本属于某一类别的概率

**核心内容**：
- **Sigmoid函数**：将线性输出转换为概率值
- **损失函数**：对数损失（Log Loss）/ 交叉熵损失
- **决策边界**：线性决策边界
- **多分类策略**：OvR（一对多）、OvO（一对一）
- **实际案例**：心脏病预测、手写数字识别

**应用场景**：
- 二分类问题（垃圾邮件检测、疾病诊断）
- 多分类问题（文本分类、图像分类）
- 概率预测（风险评估、信用评分）

📄 [查看逻辑回归技术文档](07_logistic_regression_tech_doc.md)

---

### 8. 感知机

感知机是神经网络和深度学习的起源算法，是最简单的人工神经网络模型。

**算法原理**：
- 接收多个输入信号，输出一个信号
- 通过学习调整权重和偏置

**核心内容**：
- **基本结构**：输入、权重、偏置、激活函数
- **逻辑电路实现**：与门、或门、与非门
- **感知机的局限性**：无法解决非线性问题（如异或门）
- **多层感知机**：通过组合感知机解决非线性问题

**历史意义**：
- 1957年由Frank Rosenblatt提出
- 奠定了神经网络的理论基础
- 推动了深度学习的发展

📄 [查看感知机技术文档](08_perceptron_tech_doc.md)

---

### 9. 其他监督学习算法（了解）

介绍其他常用的监督学习算法，作为扩展学习内容。

**主要内容**：
- **朴素贝叶斯**：基于贝叶斯定理的概率分类算法
- **决策树**：基于树结构的分类和回归算法
- **支持向量机（SVM）**：基于最大间隔的分类算法
- **集成学习**：随机森林、AdaBoost、GBDT等

📄 [查看其他监督学习算法技术文档](09_other_supervised_learning_tech_doc.md)

---

## 🛠️ 环境要求

- **Python**: 3.7+
- **核心库**: 
  - NumPy >= 1.19.0
  - Pandas >= 1.2.0
  - Matplotlib >= 3.3.0
  - Scikit-learn >= 0.24.0
  - Seaborn >= 0.11.0
- **开发环境**: Jupyter Notebook

## 📦 安装依赖

```bash
pip install numpy pandas matplotlib scikit-learn seaborn jupyter
```

或者使用requirements.txt（如果存在）：

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 运行Python脚本

```bash
# 基础示例
python ch02_base/1_fitting_test.py
python ch02_base/2_regularization.py

# 模型评估
python ch02_base/metrics/2_classification_report.py

# KNN算法
python ch03_knn/4_heart_disease.py

# 线性回归
python ch04_linear_regression/2_gradient_descent.py
python ch04_linear_regression/4_advertising.py

# 逻辑回归
python ch05_logistic_regression/2_heart_disease.py
python ch05_logistic_regression/3_digit_recognizer.py

# 感知机
python ch06_perceptron/1_logic_gate.py
```

### 运行Jupyter Notebook

```bash
jupyter notebook
```

然后在浏览器中打开对应的 `.ipynb` 文件进行交互式学习。

---

## 📖 学习路径建议

### 初学者路径

1. **第一阶段：基础概念**
   - 阅读 `03_model_evaluation_selection_tech_doc.md` 了解模型评估基础
   - 运行 `ch02_base/1_fitting_test.py` 理解欠拟合与过拟合

2. **第二阶段：特征工程**
   - 阅读 `01_feature_engineering_tech_doc.md`
   - 运行 `ch02_base/feature/` 下的 notebooks

3. **第三阶段：优化算法**
   - 阅读 `02_model_solving_algorithms_tech_doc.md`
   - 运行 `ch02_base/3_gradient_descent1.ipynb`

4. **第四阶段：经典算法**
   - KNN算法：`05_knn_algorithm_tech_doc.md` + `ch03_knn/`
   - 线性回归：`06_linear_regression_tech_doc.md` + `ch04_linear_regression/`
   - 逻辑回归：`07_logistic_regression_tech_doc.md` + `ch05_logistic_regression/`
   - 感知机：`08_perceptron_tech_doc.md` + `ch06_perceptron/`

5. **第五阶段：扩展学习**
   - 其他监督学习算法：`09_other_supervised_learning_tech_doc.md`

### 进阶学习

- 修改代码中的超参数，观察结果变化
- 尝试将算法应用到自己的数据集
- 深入理解数学推导过程
- 比较不同算法的优缺点

---

## 📊 代码示例详解

### 欠拟合与过拟合演示

`ch02_base/1_fitting_test.py` 展示了不同复杂度模型的拟合效果：

- **欠拟合**：模型过于简单，无法捕捉数据规律（高偏差）
- **恰好拟合**：模型复杂度适中，能够较好地拟合数据
- **过拟合**：模型过于复杂，过度拟合训练数据中的噪声（高方差）

### 正则化演示

`ch02_base/2_regularization.py` 展示了正则化如何防止过拟合：

- **无正则化**：模型容易过拟合，在测试集上表现差
- **L1正则化（Lasso）**：能够进行特征选择，将不重要特征的系数压缩为0
- **L2正则化（Ridge）**：对所有特征进行平滑处理，防止过拟合

### 心脏病预测案例

`ch03_knn/4_heart_disease.py` 完整展示了机器学习项目流程：

1. 数据加载与预处理
2. 特征工程（标准化、编码）
3. 模型训练（KNN分类器）
4. 模型评估（准确率、混淆矩阵）
5. 超参数调优（网格搜索）
6. 模型保存与加载

### 广告投放效果预测

`ch04_linear_regression/4_advertising.py` 展示了线性回归的实际应用：

1. 多特征数据的处理
2. 特征标准化
3. 多种求解方法的比较
4. 模型解释与业务洞察

### 手写数字识别案例

`ch05_logistic_regression/3_digit_recognizer.py` 展示了逻辑回归在多分类问题中的应用：

1. MNIST数据集加载
2. 多分类策略（OvR）
3. 模型训练与评估
4. 分类结果可视化

### 逻辑门实现

`ch06_perceptron/1_logic_gate.py` 展示了感知机的基本应用：

1. 与门（AND）、或门（OR）、与非门（NAND）的实现
2. 感知机权重和偏置的理解
3. 异或门（XOR）的多层感知机实现
4. 感知机的局限性演示

---

## 📚 技术文档索引

| 文档 | 内容概述 | 核心知识点 |
|------|----------|-----------|
| [特征工程技术文档](01_feature_engineering_tech_doc.md) | 特征选择和降维方法 | 方差过滤、相关系数、PCA |
| [模型求解算法技术文档](02_model_solving_algorithms_tech_doc.md) | 优化算法详解 | 梯度下降、牛顿法、拟牛顿法 |
| [模型评估和选择技术文档](03_model_evaluation_selection_tech_doc.md) | 评估指标和方法 | 损失函数、正则化、交叉验证 |
| [模型评估技术文档](04_model_evaluation_tech_doc.md) | 评估实践指南 | 混淆矩阵、ROC曲线、AUC |
| [KNN算法技术文档](05_knn_algorithm_tech_doc.md) | KNN算法详解 | 距离度量、超参数调优 |
| [线性回归技术文档](06_linear_regression_tech_doc.md) | 线性回归详解 | 正规方程、梯度下降、MLE |
| [逻辑回归技术文档](07_logistic_regression_tech_doc.md) | 逻辑回归详解 | Sigmoid函数、交叉熵、多分类 |
| [感知机技术文档](08_perceptron_tech_doc.md) | 感知机详解 | 逻辑门、多层感知机 |
| [其他监督学习算法](09_other_supervised_learning_tech_doc.md) | 扩展算法 | 朴素贝叶斯、决策树、SVM |

---

## 🤝 贡献指南

欢迎提出问题和改进建议！您可以通过以下方式参与：

1. **提交Issue**：报告bug或提出新功能建议
2. **提交Pull Request**：改进代码或文档
3. **分享经验**：将学习心得分享给社区

---

## 📝 许可证

本项目仅供学习使用，遵循MIT许可证。

---

## 🙏 致谢

感谢所有为机器学习开源社区做出贡献的研究者和开发者。

---

**Happy Learning! 🎉**

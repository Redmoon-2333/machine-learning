# 机器学习基础代码库

这是一个机器学习基础学习项目，涵盖了机器学习流程中的核心环节，包括特征工程、模型评估与选择、模型求解算法等关键内容。项目通过理论文档和实际代码相结合的方式，帮助学习者深入理解机器学习的核心概念和实现方法。

## 项目结构

```
.
├── ch02_base/                    # 第二章基础代码
│   ├── feature/                  # 特征工程
│   │   ├── 1_variance_filter.ipynb      # 低方差过滤
│   │   ├── 2_pearson.ipynb              # 皮尔逊相关系数
│   │   ├── 3_spearman.ipynb             # 斯皮尔曼相关系数
│   │   └── 4_pca.ipynb                  # PCA降维
│   ├── 1_fitting_test.py         # 欠拟合/过拟合演示
│   ├── 2_regularization.py       # 正则化演示
│   ├── 3_gradient_descent1.ipynb # 梯度下降法（一）
│   └── 4_gradient_descent2.ipynb # 梯度下降法（二）
├── data/                         # 数据集
│   └── advertising.csv           # 广告数据集
├── feature_engineering_tech_doc.md              # 特征工程技术文档
├── model_evaluation_selection_tech_doc.md      # 模型评估和选择技术文档
└── model_solving_algorithms_tech_doc.md        # 模型求解算法技术文档
```

## 主要内容

### 1. 特征工程

特征工程是机器学习流程中的关键环节，包括特征选择和降维两个核心部分。

- **低方差过滤法**：基于特征的方差进行过滤，适用于快速去除无信息特征
- **皮尔逊相关系数法**：基于线性相关关系选择特征，适用于连续型变量且存在线性关系的场景
- **斯皮尔曼相关系数法**：基于单调相关关系选择特征，适用于非线性关系或有序分类变量的场景
- **PCA降维**：通过线性变换将高维数据映射到低维空间，适用于高维数据可视化和特征提取

详细内容请参考：[特征工程技术文档](feature_engineering_tech_doc.md)

### 2. 模型评估和选择

模型评估和选择是机器学习流程中的关键环节，直接影响到模型的性能和泛化能力。

- **损失函数**：衡量模型预测值与真实值之间的差异，包括MSE、MAE、交叉熵损失等
- **经验误差与泛化误差**：理解模型在训练数据和未见数据上的表现差异
- **欠拟合与过拟合**：模型复杂度与数据拟合程度的关系
- **正则化**：通过L1和L2正则化控制模型复杂度，防止过拟合
- **交叉验证**：利用有限数据评估模型泛化能力，包括k折交叉验证等方法

详细内容请参考：[模型评估和选择技术文档](model_evaluation_selection_tech_doc.md)

### 3. 模型求解算法

模型求解算法是机器学习流程中的核心环节，负责在给定模型结构和训练数据的情况下，找到最优的模型参数。

- **梯度下降法**：
  - 批量梯度下降（BGD）
  - 随机梯度下降（SGD）
  - 小批量梯度下降（Mini-batch GD）
  - 进阶优化策略：动量法、AdaGrad、RMSprop、Adam等
- **解析解法**：通过数学推导直接求解最优参数，适用于线性回归等模型
- **牛顿法和拟牛顿法**：利用二阶导数信息加速收敛，包括BFGS、L-BFGS等

详细内容请参考：[模型求解算法技术文档](model_solving_algorithms_tech_doc.md)

## 环境要求

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Seaborn
- Jupyter Notebook

## 安装依赖

```bash
pip install numpy matplotlib pandas scikit-learn seaborn jupyter
```

## 使用说明

### 运行Python脚本

```bash
python ch02_base/1_fitting_test.py
python ch02_base/2_regularization.py
```

### 运行Jupyter Notebook

```bash
jupyter notebook
```

然后在浏览器中打开对应的 `.ipynb` 文件。

## 代码示例

### 欠拟合与过拟合演示

[1_fitting_test.py](ch02_base/1_fitting_test.py) 展示了不同复杂度模型的拟合效果：

- 欠拟合：模型过于简单，无法捕捉数据中的规律
- 恰好拟合：模型复杂度适中，能够较好地拟合数据
- 过拟合：模型过于复杂，过度拟合训练数据中的噪声

### 正则化演示

[2_regularization.py](ch02_base/2_regularization.py) 展示了正则化如何防止过拟合：

- 无正则化：模型容易过拟合
- L1正则化（Lasso）：能够进行特征选择，将不重要特征的系数压缩为0
- L2正则化（Ridge）：对所有特征进行平滑处理，防止过拟合

## 学习建议

1. **先阅读技术文档**：理解每个算法的原理和数学基础
2. **运行代码示例**：通过实际运行代码加深理解
3. **修改参数实验**：尝试修改超参数，观察结果变化
4. **应用到自己的数据**：将学到的知识应用到实际项目中

## 技术文档索引

- [特征工程技术文档](feature_engineering_tech_doc.md) - 详细介绍特征选择和降维方法
- [模型评估和选择技术文档](model_evaluation_selection_tech_doc.md) - 详细介绍模型评估指标和方法
- [模型求解算法技术文档](model_solving_algorithms_tech_doc.md) - 详细介绍各种优化算法

## 许可证

本项目仅供学习使用。

## 贡献

欢迎提出问题和改进建议！

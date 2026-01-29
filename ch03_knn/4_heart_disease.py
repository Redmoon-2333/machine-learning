import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib
#加载数据集
data = pd.read_csv('../data/heart.csv')

# 处理缺失值
data = data.dropna()

data.info()
print(data.head())

# 数据集划分

# 划分特征和标签
X = data.drop('目标', axis=1)
y = data['目标']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 特征工程

# 类别型特征
categorical_features = ["胸痛类型", "静息心电图", "ST段斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动诱发心绞痛"]

# 创建列转换器
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

print(f"转换后的训练集形状: {x_train.shape}")
print(f"转换后的测试集形状: {x_test.shape}")

# # 创建模型
# knn = KNeighborsClassifier(n_neighbors=3)
#
# # 训练模型
# knn.fit(x_train, y_train)
#
# # 模型评估，计算预测准确率
# accuracy = knn.score(x_test, y_test)
# print(f"模型准确率: {accuracy:.5f}")

# 保存模型
# joblib.dump(knn, 'heart_disease_model.pkl')

# 加载模型，对新数据进行预测
# knn_load = joblib.load('heart_disease_model.pkl')
# print(f"预测类别：{knn_load.predict(x_test[10:11])}, 真实类别：{y_test[10]}")

# 创建knn分类器
knn = KNeighborsClassifier()

# 定义网格搜索参数列表
param_grid = {
    'n_neighbors': list(range(1, 11)),
    'weights': ['uniform', 'distance'],
}

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
print(knn.score(x_test, y_test))







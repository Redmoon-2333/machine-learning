import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# 1. 加载数据集
data = pd.read_csv('../data/heart.csv')
data = data.dropna()

# 2. 划分数据集
X = data.drop('目标', axis=1)
y = data['目标']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程
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

# 4. 模型定义和训练
model = LogisticRegression()
model.fit(x_train, y_train)

# 5. 准确率计算，评估模型
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy:.4f}")



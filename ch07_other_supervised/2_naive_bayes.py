# 朴素贝叶斯分类示例
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report


# 1. 高斯朴素贝叶斯（适用于连续特征）
print("=" * 50)
print("高斯朴素贝叶斯 - 鸢尾花分类")
print("=" * 50)

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建模型并训练
model = GaussianNB()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"各类别先验概率: {model.class_prior_}")


# 2. 多项式朴素贝叶斯（适用于文本分类）
print("\n" + "=" * 50)
print("多项式朴素贝叶斯 - 文本分类示例")
print("=" * 50)

# 简单文本分类示例
texts = [
    "I love this movie, it is great",
    "This film is wonderful and amazing",
    "Terrible movie, I hate it",
    "Bad film, very boring",
    "Great story and excellent acting",
    "Worst movie ever, do not watch"
]
labels = [1, 1, 0, 0, 1, 0]  # 1=正面, 0=负面

# 文本向量化
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

# 训练多项式朴素贝叶斯
nb_model = MultinomialNB()
nb_model.fit(X_text, labels)

# 预测新文本
new_texts = ["This movie is great", "I hate this terrible film"]
X_new = vectorizer.transform(new_texts)
predictions = nb_model.predict(X_new)

for text, pred in zip(new_texts, predictions):
    sentiment = "正面" if pred == 1 else "负面"
    print(f"'{text}' -> {sentiment}")

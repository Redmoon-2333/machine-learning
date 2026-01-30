import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 1. 生成数据
X ,y =make_blobs(n_samples=300, centers=3,cluster_std=2)

# 2. 定义模型并聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 3. 获取聚类结果
centers = kmeans.cluster_centers_

# 4. 得到所有样本点的分类标签
y_pred = kmeans.predict(X)
print(y_pred)
plt.scatter(X[:,0],X[:,1],c=y_pred,label="聚类结果")
plt.scatter(centers[:,0],centers[:,1],c='red',label="质心")
plt.legend()

plt.show()

# 打印评价指标
print("簇内平方和:", kmeans.inertia_)
print("轮廓系数:", silhouette_score(X, y_pred))
print("Calinski-Harabasz指数:", calinski_harabasz_score(X, y_pred))




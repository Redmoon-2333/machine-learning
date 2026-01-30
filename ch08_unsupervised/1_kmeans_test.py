import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 1. 生成数据
X ,y =make_blobs(n_samples=300, centers=3,cluster_std=2)

# 画出散点图
fig,ax=plt.subplots(2,figsize=(8,8))
ax[0].scatter(X[:,0],X[:,1],c=y,label="原始数据")
ax[0].set_title("原始数据")
ax[0].legend()

# 2. 定义模型并聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 3. 获取聚类结果
centers = kmeans.cluster_centers_

# 4. 得到所有样本点的分类标签
y_pred = kmeans.predict(X)
print(y_pred)
ax[1].scatter(X[:,0],X[:,1],c=y_pred,label="聚类结果")
ax[1].scatter(centers[:,0],centers[:,1],c='red',label="质心")
ax[1].set_title("K-means聚类结果")
ax[1].legend()
plt.show()





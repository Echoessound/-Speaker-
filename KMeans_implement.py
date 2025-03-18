'''
import numpy as np
from sklearn.cluster import KMeans

# 定义几个点
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# 初始化KMeans聚类模型
kmeans = KMeans(n_clusters=2, random_state=0)
# 使用数据训练模型
kmeans.fit(X)

# 输出聚类结果
print(kmeans.labels_)

# 输出模型的聚类中心点
print(kmeans.cluster_centers_)
'''

import numpy as np
import matplotlib.pyplot as plt


# 我们自己去实现KMeans算法,不依赖任何机器学习库
def kmeans(X, k, max_iter=100):
    # 随机初始化K个中心点
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    # 这个labels集合我们就存放KMeans给样本打的标签
    labels = np.zeros(X.shape[0])

    for i in range(max_iter):
        # 分配样本到最近的中心点
        # 求的是欧式距离
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        # 看每一个样本离哪个中心点更近，也就是距离更小
        new_labels = np.argmin(distances, axis=0)

        # 更新中心点
        for j in range(k):
            centers[j] = X[new_labels == j].mean(axis=0)

        # 如果聚类结果没有变化，则提前结束迭代
        if (new_labels == labels).all():
            break
        else:
            labels = new_labels
    return labels, centers


# 生成数据集，生成了3个组的数据
X = np.vstack((np.random.randn(100, 2) * 0.75 + np.array([1, 0]),
               np.random.randn(100, 2) * 0.25 + np.array([-0.5, 0.5]),
               np.random.randn(100, 2) * 0.5 + np.array([-0.5, -0.5])))

# 运行KMeans聚类方法
labels, centers = kmeans(X, k=3)

# 可视化聚类结果
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], marker='x', s=200, linewidths=3, color='r')
plt.show()

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载 20newsgroups 数据集
newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# 创建一个pipeline，用于文件特征提取，接着使用逻辑回归
pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=3000))

# 使用训练集训练模型
pipeline.fit(newsgroups_train.data, newsgroups_train.target)

# 在测试集上面预测
y_pred = pipeline.predict(newsgroups_test.data)

# 输出模型的准确率
print("Accuracy: %.2f" % accuracy_score(newsgroups_test.target, y_pred))

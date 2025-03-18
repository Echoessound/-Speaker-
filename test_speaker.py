import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from speaker_features import extract_features
import warnings

warnings.filterwarnings("ignore")
import time

# 音频数据所存放的目录
source = 'D:/data/speaker-identification/development_set/'
# 模型文件存放的目录
modelpath = "./speaker_models/"
# 测试集文件的名称
test_file = 'D:/data/speaker-identification/development_set_test.txt'
file_paths = open(test_file, 'r')
# 得到每一个模型文件的路径
gmm_files = [os.path.join(modelpath, fname) for fname in
             os.listdir(modelpath) if fname.endswith('.gmm')]
# 通过路径加载每一个模型文件
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
# 获取每一个人的人名
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
            in gmm_files]

for path in file_paths:

    path = path.strip()
    # 读取每一个测试音频
    sr, audio = read(source + path)
    # 进行特征的提取，每一个对应40个维度
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    # 使用模型
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)

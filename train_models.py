import numpy as np
from scipy.io.wavfile import read
from speaker_features import extract_features
from sklearn.mixture import GaussianMixture
import pickle

source = 'D:/data/speaker-identification/development_set/'
train_file = 'D:/data/speaker-identification/development_set_enroll.txt'

dest = './speaker_models/'

file_paths = open(train_file, 'r')

features = np.asarray(())

count = 1

for path in file_paths:
    path = path.strip()
    print(source + path)

    # 一个个地读取了原声音文件
    sr, audio = read(source + path)

    # 提取40个维度特征（MFCC+ΔMFCC）
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    # 训练集是每5个文件对应一个人
    if count == 5:
        # 这里就相对于是用一个人的5条语音文件，对应来使用一个GMM模型建模
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # 落地保持每一个人对应的GMM模型
        picklefile = path.split('-')[0] + '.gmm'
        pickle.dump(gmm, open(dest+picklefile, 'wb'))

        features = np.asarray(())
        count = 0
    count = count + 1

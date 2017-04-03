import numpy as np
import matplotlib.pylab as plt

#### 活性化関数 ####
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a)
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


#### 損失関数 ####
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorだった場合、正解ラベルのインデックスに変換
    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

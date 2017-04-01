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
    # 微小な値を加算してlog(0)でマイナス無限大となることを防ぐ
    delta = 1e-7
    return  -np.sum(t * np.log(y + delta))

import numpy as np
import matplotlib.pylab as plt
from sigmoid import sigmoid

# 恒等関数
def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # weight
    network['b1'] = np.array([0.1, 0.2, 0.3]) # bias
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

def forword(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(X, W1) + b1 # output
    Z1 = sigmoid(a1)
    a2 = np.dot(Z1, W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2, W3) + b3

    # 出力層では一般的に隠れ層と異なる活性化関数を使用する
    y = identity_function(A3)
    return y

x = np.array([1.0, 0.5])
y = forword(init_network(), forword(x))
print(Y)

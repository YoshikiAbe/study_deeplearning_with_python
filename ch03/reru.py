import numpy as np
import matplotlib.pylab as plt

def reru(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = reru(x)
plt.plot(x, y)
plt.ylim(-1, 6) # y軸の範囲を指定
plt.show()

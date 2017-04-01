import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import mean_squared_error, cross_entropy_error

# 「2」を正解とする
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 例1: 「2」の確率が最も高い場合(0.6)
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error1 = mean_squared_error(np.array(y1), np.array(t))
cross_entropy_error1 = cross_entropy_error(np.array(y1), np.array(t))
print('二条誤差(2が正解の場合): ', mean_squared_error1) # 0.097500000000000031
print('交差エントロピー誤差(2が正解の場合): ', cross_entropy_error1) # 0.097500000000000031


# 例2:「7」の確率が最も高い場合(0.6)
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error2 = mean_squared_error(np.array(y2), np.array(t))
cross_entropy_error2 = cross_entropy_error(np.array(y2), np.array(t))
print('二条誤差(2が正解の場合): ', mean_squared_error2) # 0.59750000000000003
print('交差エントロピー誤差(2が正解の場合): ', cross_entropy_error2) # 0.097500000000000031

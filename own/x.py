


import numpy as np

import tensorflow as tf

class FNV:
    def __init__(self):
        pass

    def infer( self, trajectories ):
        pass

    def train( self, trajectories ):
        pass








from scipy.signal import lfilter

x = np.array([1,10,100,1000,10000,100000,1000000])

g = 0.9
res = list()

for since in range(len(x)):
    s = 0
    for i, v in enumerate(x[since:]):
        s += v * np.power(g,i)
    res.append(s)

print( x[::-1])
print(res)
print( lfilter([1],[1,-0.9],x[::-1],axis=0)[::-1])

exit()





















import numpy as np
import tensorflow as tf

np.var
# 动作空间
#       如果是一个N维

def build():
    pass


if __name__ == '__main__':
    exit()

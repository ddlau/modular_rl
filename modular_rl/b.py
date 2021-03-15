import numpy as np

import torch

from torch import nn

from scipy.signal import lfilter


def discount(x, decay, check=None):
    y = lfilter([1], [1, -decay], x[::-1], axis=0)[::-1]

    if check:
        v = 0
        for t in reversed(range(len(x))):
            assert y[t] == (v := v * decay + x[t])

    return y


####












class Model( nn.Module):
    def __init__(self, len_of_input, len_of_output):
        super().__init__()

        self.linear1st = nn.Linear( len_of_input, 64 )

        self.linear1st.parameters()
        self.linear2nd = nn.Linear( 64,64)




def tst():
    x = nn.Linear(3,5)

    for i in x.parameters():
        print(type(i), i.size())

    for n, v in x.named_parameters():
        print( n )

    a = np.arange( x.weight.numel() ).reshape( x.weight.size())
    x.weight.data.copy_( torch.as_tensor(   a  ))

    a[2,2]=1000
    print( f'x.out_features:{x.out_features}')
    print( x.weight, type(x.weight))
    x = Model(10,30)
    print(x)


tst()
if __name__ == '__main__':
    exit()

import numpy as np
import torch

def seq2seq_softmax_with_mask(entries, mask):
    entries = entries[:, :, 0]
    maxs = entries.max(1, keepdim=True)[0]
    # print(entries.shape, maxs.shape, mask.shape)
    entries = torch.exp(entries - maxs) * mask
    return entries / (entries.sum(dim=1, keepdim=True) + 1e-15)


def batch_iterator(X, Y=None, batch_size=32, synchronize=True, length=None):
    assert Y is None or not synchronize or X.shape[0] == Y.shape[0]
    if length is None:
        length = X.shape[0]
    if synchronize:
        indx = indy = np.random.choice(X.shape[0], size=length)
    else:
        indx = np.random.choice(X.shape[0], size=length)
        indy = np.random.choice(Y.shape[0], size=length)
    for i in range(0, length, batch_size):
        if Y is not None:
            yield X[indx[i:i + batch_size]], Y[indy[i:i + batch_size]]
        else:
            yield X[indx[i:i + batch_size]]


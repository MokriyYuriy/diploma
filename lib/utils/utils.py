import numpy as np
import torch

def seq2seq_softmax_with_mask(entries, mask):
    entries = entries[:, :, 0]
    maxs = entries.max(1, keepdim=True)[0]
    # print(entries.shape, maxs.shape, mask.shape)
    entries = torch.exp(entries - maxs) * mask
    return entries / (entries.sum(dim=1, keepdim=True) + 1e-15)


def batch_iterator(X, Y=None, batch_size=32):
    assert Y is None or X.shape[0] == Y.shape[0]
    ind = np.arange(X.shape[0])
    np.random.shuffle(ind)
    for i in range(0, X.shape[0], batch_size):
        if Y is not None:
            yield X[ind[i:i + batch_size]], Y[ind[i:i + batch_size]]
        else:
            yield X[ind[i:i + batch_size]]


def load_pair_dataset(filename, alph1, alph2):
    x, y = [], []
    with open(filename, 'r') as ftr:
        for line in ftr:
            try:
                word1, word2 = line.split()
            except ValueError:
                continue
            x.append(alph1.letter2index(word1))
            y.append(alph2.letter2index(word2))
    return np.array(x), np.array(y)
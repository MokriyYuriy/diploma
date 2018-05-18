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

def inplace_clip_gradient(model, max_norm=1.0):
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.data = param.grad.data.clamp(-max_norm, max_norm)

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    """
    Just a copy of function from newer version of pytorch
    """

    if max_norm is None:
        return

    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef.item())
    return total_norm

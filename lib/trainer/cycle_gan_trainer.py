import torch
from torch.autograd import Variable

from ..utils import batch_iterator


def train_cycle_gan(model, train_X, train_Y, val_src_words, val_trg_words, metrics=dict(), n_epochs=5, use_cuda=False):
    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(batch_iterator(train_X, train_Y, synchronize=False)):
            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(y))
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            model.compute_losses(x, reversed=False)
            model.compute_losses(y, reversed=True)




import time
import os

import torch
from torch.autograd import Variable

from ..utils import batch_iterator
from ..loss import cross_entropy
from ..metrics import compute_bleu_score

def train_generator(model, opt, alph_Y, train_X, train_Y, val_src_words, val_trg_words,
    checkpoints_folder, metrics_compute_freq=50, n_epochs=7):

    cur_loss = 0
    for epoch in range(n_epochs):
        model.train()
        start_time = time.time()
        for i, (x, y) in enumerate(batch_iterator(train_X, train_Y)):
            inputs = Variable(torch.from_numpy(x))
            targets = Variable(torch.from_numpy(y))
            log_predictions = model(inputs, targets)
            # print(x)
            loss = cross_entropy(log_predictions, targets[:, 1:].contiguous(), alph_Y)
            # print(loss.data, log_predictions.data.min())
            loss.backward()
            cur_loss = 0.9 * cur_loss + 0.1 * loss.data[0]
            opt.step()
            opt.zero_grad()
            if (i + 1) % metrics_compute_freq == 0:
                print("epoch: {} iter: {} loss: {}".format(epoch, i, cur_loss))
        model.eval()
        val_score = compute_bleu_score(model, val_src_words, val_trg_words)
        print("epoch: {} val_score: {} time: {}"
              .format(epoch, val_score, time.time() - start_time))
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_folder, "state_dict_{}_{}.pth".format(epoch, val_score)))

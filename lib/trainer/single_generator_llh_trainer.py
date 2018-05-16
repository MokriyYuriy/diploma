import time
import os

import torch
from torch.autograd import Variable

from ..utils import batch_iterator, build_history, update_history, plot_history
from ..loss import cross_entropy
from ..metrics import compute_bleu_score

def train_generator(model, opt, alph_Y, train_X, train_Y, val_src_words, val_trg_words,
    checkpoints_folder, metrics_compute_freq=50, n_epochs=7, use_cuda=False):
    history = build_history([
        ("cross_entropy", dict(smoothed=True, xlabel="iterations")),
        ("bleu", dict(smoothed=False, xlabel="epochs"))
    ])
    previous_epoch_bleu = None
    previous_epoch_time = None
    for epoch in range(n_epochs):
        model.train()
        start_time = time.time()
        for i, (x, y) in enumerate(batch_iterator(train_X, train_Y)):
            inputs = Variable(torch.from_numpy(x))
            targets = Variable(torch.from_numpy(y))
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            log_predictions = model(inputs, targets)
            # print(x)
            loss = cross_entropy(log_predictions, targets[:, 1:].contiguous(), alph_Y, use_cuda)
            # print(loss.data, log_predictions.data.min())
            loss.backward()
            cur_loss = 0.9 * cur_loss + 0.1 * loss.data[0]
            update_history(history, dict(cross_entropy=loss.data[0]))
            opt.step()
            opt.zero_grad()
            if i % metrics_compute_freq + 1 == metrics_compute_freq:
                print("epoch: {} iter: {} loss: {} prev_epoch_bleu: {} prev_epoch_time"
                      .format(epoch, i, cur_loss, previous_epoch_bleu, previous_epoch_time))
                plot_history(history)

        model.eval()
        previous_epoch_bleu = compute_bleu_score(model, val_src_words, val_trg_words)
        update_history(history, dict(bleu=previous_epoch_bleu))
        previous_epoch_time = time.time() - start_time
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_folder, "state_dict_{}_{}.pth".format(epoch, previous_epoch_bleu)))

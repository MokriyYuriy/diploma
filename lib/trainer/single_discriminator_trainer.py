import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import batch_iterator
from ..loss import disc_loss


def train_discriminator(disc_model, gen_model, opt, train_X, train_Y, n_epochs=50, use_cuda=False):
    cur_loss = 0
    for epoch in range(n_epochs):
        disc_model.train()
        gen_model.eval()
        start_time = time.time()
        for i, (x, y) in enumerate(batch_iterator(train_X, train_Y)):
            inputs = Variable(torch.from_numpy(x))
            targets = Variable(torch.from_numpy(y))
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            real_data_pred = disc_model(targets)
            #print(targets.shape, gen_model.translate(inputs).shape)
            gen_data_pred = disc_model(gen_model.translate(inputs).detach())
            #print(targets, gen_model.translate(inputs), real_data_pred, gen_data_pred)
            #print(gen_data_pred)
            #print(real_data_pred)
            loss = disc_loss(real_data_pred, gen_data_pred)
            cur_loss = 0.9 * cur_loss + 0.1 * loss.data[0]
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 10 == 9:
                print(loss.data[0])
            #break
        print(cur_loss)
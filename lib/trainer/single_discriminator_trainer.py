import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import batch_iterator


def train_discriminator(disc_model, gen_model, opt, train_X, train_Y, n_epochs=50):
    cur_loss = 0
    for epoch in range(n_epochs):
        disc_model.train()
        gen_model.eval()
        start_time = time.time()
        for i, (x, y) in enumerate(batch_iterator(train_X, train_Y)):
            inputs = Variable(torch.from_numpy(x))
            targets = Variable(torch.from_numpy(y))
            real_data_pred = disc_model(targets)
            #print(targets.shape, gen_model.translate(inputs).shape)
            gen_data_pred = disc_model(gen_model.translate(inputs))
            #print(targets, gen_model.translate(inputs), real_data_pred, gen_data_pred)
            #print(gen_data_pred)
            #print(real_data_pred)
            loss = F.binary_cross_entropy(gen_data_pred, torch.zeros_like(gen_data_pred)) \
                    + F.binary_cross_entropy(real_data_pred, torch.ones_like(real_data_pred))
            cur_loss = 0.9 * cur_loss + 0.1 * loss.data[0]
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 10 == 9:
                print(loss.data[0])
            #break
        print(cur_loss)
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import batch_iterator, build_history, update_history, plot_history
from ..loss import disc_loss


def train_discriminator(disc_model, gen_model, opt, train_X, train_Y, n_epochs=50, update_plot_freq=50, use_cuda=False):
    history = build_history([("disc_loss", dict())])
    for epoch in range(n_epochs):
        disc_model.train()
        gen_model.eval()
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
            update_history(history, dict(disc_loss=loss.data[0]))
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % update_plot_freq + 1 == update_plot_freq:
                plot_history(history)
            #break
    return history

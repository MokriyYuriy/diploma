import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import batch_iterator, build_history, update_history, plot_history, inplace_clip_gradient
from ..loss import disc_cross_entropy


def train_discriminator(
    disc_model, gen_model, opt, train_X, train_Y, n_epochs=50,
    update_plot_freq=50, alpha=0.01, clipping=1.0, use_cuda=False
):
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
            gen_data_pred = disc_model(gen_model.translate(inputs, strategy='sampling').detach())
            #print(targets, gen_model.translate(inputs), real_data_pred, gen_data_pred)
            #print(gen_data_pred)
            #print(real_data_pred)
            loss = disc_cross_entropy(real_data_pred, gen_data_pred, alpha=alpha)
            update_history(history, dict(disc_loss=loss.data[0]))
            loss.backward()
            inplace_clip_gradient(disc_model, clipping)
            opt.step()
            opt.zero_grad()
            if i % update_plot_freq + 1 == update_plot_freq:
                plot_history(history)
            #break
    return history

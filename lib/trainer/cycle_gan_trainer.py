from torch.autograd import Variable

from ..loss import policy_gradient_loss
from ..utils import batch_iterator


def train_cycle_gan(model, train_X, train_Y, val_src_words, val_trg_words, metrics=dict(), n_epochs=5):
    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(batch_iterator(train_X, train_Y)):
            x = Variable(x)
            y = Variable(y)
            x_disc_predictions, x_result_sequence, x_logits, x_baselline_disc_predictions, x_reversed_logits \
                = model(x, reversed=False)
            y_disc_predictions, y_result_sequence, y_logits, y_baselline_disc_predictions, y_reversed_logits \
                = model(y, reversed=True)
            loss = policy_gradient_loss()




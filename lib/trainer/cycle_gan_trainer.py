import os

import torch
from torch.autograd import Variable

from ..loss import cross_entropy
from ..metrics import compute_corpus_bleu_score, compute_cycle_corpus_bleu_score
from ..utils import batch_iterator, build_history, plot_history, update_history, clip_grad_norm_


def train_cycle_gan(
    model, backprop_opt, rl_opt, paired_X, paired_Y, single_X, single_Y, val_src_words, val_trg_words,
    n_epochs=5, samples_per_epoch=10000, batch_size=32, max_norm=None, use_cuda=False, coef=None, update_plot_freq=50,
    checkpoints_folder=None
):
    history = build_history([
        ('src_supervised_ce', dict()),
        ('trg_supervised_ce', dict()),
        ('src_bleu_score', dict(xlabel="epochs", smoothed=False)),
        ('trg_bleu_score', dict(xlabel="epochs", smoothed=False)),
        ('src_cycle_bleu_score', dict(xlabel="epochs", smoothed=False)),
        ('trg_cycle_bleu_score', dict(xlabel="epochs", smoothed=False)),
        ('src_disc_advantage', dict()),
        ('trg_disc_advantage', dict()),
        ('src_disc_reward', dict()),
        ('trg_disc_reward', dict()),
        ('src_disc_loss', dict()),
        ('trg_disc_loss', dict()),
        ('src_cycle_loss', dict()),
        ('trg_cycle_loss', dict()),
        ('src_entropy', dict()),
        ('trg_entropy', dict()),
        ('backprop_grad_norm', dict()),
        ('rl_grad_norm', dict())
    ])

    model.eval()
    src_epoch_bleu = compute_corpus_bleu_score(model.src_gan.gen_model, val_src_words, val_trg_words)
    trg_epoch_bleu = compute_corpus_bleu_score(model.trg_gan.gen_model, val_trg_words, val_src_words)
    update_history(history, dict(
        src_bleu_score=src_epoch_bleu,
        trg_bleu_score=trg_epoch_bleu
    ))


    for epoch in range(n_epochs):
        model.train()
        for i, (single_batch, paired_batch) in enumerate(zip(
            batch_iterator(single_X, single_Y, batch_size=batch_size, synchronize=False, length=samples_per_epoch),
            batch_iterator(paired_X, paired_Y, batch_size=batch_size, length=samples_per_epoch)
        )):

            single_x = Variable(torch.from_numpy(single_batch[0]))
            single_y = Variable(torch.from_numpy(single_batch[1]))
            paired_x = Variable(torch.from_numpy(paired_batch[0]))
            paired_y = Variable(torch.from_numpy(paired_batch[1]))

            if use_cuda:
                single_x = single_x.cuda()
                single_y = single_y.cuda()
                paired_x = paired_x.cuda()
                paired_y = paired_y.cuda()

            x_pg_disc_loss, x_pg_cycle_loss, y_true_disc_loss, x_fake_disc_loss, x_cycle_ce, x_pg_entropy, x_advantages \
                = model.compute_losses(single_x, reversed=False)
            y_pg_disc_loss, y_pg_cycle_loss, x_true_disc_loss, y_fake_disc_loss, y_cycle_ce, y_pg_entropy, y_advantages \
                = model.compute_losses(single_y, reversed=True)

            x_disc_loss = x_fake_disc_loss + x_true_disc_loss
            y_disc_loss = y_fake_disc_loss + y_true_disc_loss

            x_sv_log_pred = model.src_gan.gen_model(paired_x, paired_y)
            y_sv_log_pred = model.trg_gan.gen_model(paired_y, paired_x)

            x_sv_loss = cross_entropy(
                x_sv_log_pred, paired_y[:, 1:].contiguous(),
                model.trg_gan.gen_model.encoder.alphabet, reduce_mean=False
            ).mean()
            y_sv_loss = cross_entropy(
                y_sv_log_pred, paired_x[:, 1:].contiguous(),
                model.src_gan.gen_model.encoder.alphabet, reduce_mean=False
            ).mean()

            update_history(history, dict(
                src_supervised_ce=x_sv_loss.data[0],
                trg_supervised_ce=y_sv_loss.data[0],
                src_disc_advantage=x_advantages['disc_advantage'],
                trg_disc_advantage=y_advantages['disc_advantage'],
                src_disc_reward=x_advantages['disc_reward'],
                trg_disc_reward=x_advantages['disc_reward'],
                src_disc_loss=x_disc_loss.data[0],
                trg_disc_loss=y_disc_loss.data[0],
                src_cycle_loss=x_cycle_ce.data[0],
                trg_cycle_loss=y_cycle_ce.data[0],
                src_entropy=x_advantages['entropy'],
                trg_entropy=y_advantages['entropy']
            ))

            sv_loss = x_sv_loss + y_sv_loss
            disc_loss = x_disc_loss + y_disc_loss
            cycle_ce = x_cycle_ce + y_cycle_ce
            pg_disc_loss = x_pg_disc_loss + y_pg_disc_loss
            pg_cycle_loss = x_pg_cycle_loss + y_pg_cycle_loss
            pg_entropy = x_pg_entropy + y_pg_entropy


            backprop_loss = coef['disc_loss'] * disc_loss + coef['cycle_ce'] * cycle_ce \
                            + coef['supervised_ce'] * sv_loss

            rl_loss = coef['pg_disc_loss'] * pg_disc_loss \
                      + coef['pg_cycle_loss'] * pg_cycle_loss \
                      + coef['pg_entropy'] * pg_entropy

            backprop_loss.backward()
            backrop_grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
            backprop_opt.step()
            backprop_opt.zero_grad()

            rl_loss.backward()
            rl_grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
            rl_opt.step()
            rl_opt.zero_grad()

            update_history(history, dict(
                backprop_grad_norm=backrop_grad_norm,
                rl_grad_norm=rl_grad_norm
            ))

            if i % update_plot_freq + 1 == update_plot_freq:
                plot_history(history)

        model.eval()
        src_epoch_bleu = compute_corpus_bleu_score(model.src_gan.gen_model, val_src_words, val_trg_words)
        trg_epoch_bleu = compute_corpus_bleu_score(model.trg_gan.gen_model, val_trg_words, val_src_words)
        src_epoch_cycle_bleu = compute_cycle_corpus_bleu_score(
            model.src_gan.gen_model, model.trg_gan.gen_model, val_src_words
        )
        trg_epoch_cycle_bleu = compute_cycle_corpus_bleu_score(
            model.trg_gan.gen_model, model.src_gan.gen_model, val_trg_words
        )
        update_history(history, dict(
            src_bleu_score=src_epoch_bleu,
            trg_bleu_score=trg_epoch_bleu,
            src_cycle_bleu_score=src_epoch_cycle_bleu,
            trg_cycle_bleu_score=trg_epoch_cycle_bleu
        ))

        plot_history(history)
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_folder, "state_dict_{}_{}_{}.pth".format(
                       epoch, src_epoch_bleu, trg_epoch_bleu)))
    return history

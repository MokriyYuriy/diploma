import torch
import torch.nn.functional as F


def disc_cross_entropy(real_logits, fake_logits, alpha=0.01, sep_return=False):
    real_part = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) \
                + alpha * (real_logits ** 2).mean()
    fake_part = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)) \
                + alpha * (fake_logits ** 2).mean()
    if sep_return:
        return real_part, fake_part
    else:
        return real_part + fake_part
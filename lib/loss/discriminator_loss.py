import torch
import torch.nn.functional as F


def disc_cross_entropy(real_logits, fake_logits, sep_return=False):
    if sep_return:
        return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)),\
               F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    else:
        return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) \
               + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
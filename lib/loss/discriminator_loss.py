import torch
import torch.nn.functional as F


def disc_loss(self, real_logits, fake_logits):
    return F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) \
           + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
import torch.nn.functional as F


def policy_loss(advantages, logits, mask):
    """
    Compute such function that its gradient is policy gradient.
    """

    policy_term = (logits * mask).sum(1).sum(1)
    return -(policy_term * advantages.detach()).mean()
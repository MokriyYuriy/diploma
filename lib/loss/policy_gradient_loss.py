import torch.nn.functional as F


def policy_loss(advantages, logits, mask):
    """
    Compute such function that its gradient is policy gradient.

    :param disc_predictions: N - dicriminator outputs (log sigmoid)
    :param logits: NxTxU - log probabillities of each token in each position produced by generator
    :param result_sequence: NxT - sampled sequence from given logits distribution
    :param alphabet: Alphabet - result_sequence alphabet (it needs to obtain mask of result_sequence)
    :return:
    """

    policy_term = (logits * mask).sum(1).sum(1)
    return -(policy_term * advantages.detach()).mean()
import torch


def cross_entropy(log_predictions, targets, alphabet):
    """ Cross entropy loss for sequences
    Parameters
    ---------
    log_predictions: Tensor NxTxH
        Log probabilities
    targets: Tensor NxT
        True index-encoded translations
    alphabet: Alphabet
        Alphabet object

    """
    length_mask = alphabet.get_mask(targets)
    targets_mask = torch.zeros_like(log_predictions).scatter_(2, targets.view(*targets.shape, 1), 1.0)
    mask = targets_mask * length_mask.view(*length_mask.shape, 1)
    # print(mask.sum(1, keepdim=True).sum(2, keepdim=True))
    return (log_predictions * mask / (mask.sum(2, keepdim=True).sum(1, keepdim=True) * -log_predictions.size(0))).sum()

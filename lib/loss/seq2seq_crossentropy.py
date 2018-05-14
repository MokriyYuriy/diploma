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
    mask = alphabet.get_mask_for_3D_array(log_predictions, targets)
    # print(mask.sum(1, keepdim=True).sum(2, keepdim=True))
    return (log_predictions * mask / (mask.sum(2, keepdim=True).sum(1, keepdim=True) * -log_predictions.size(0))).sum()

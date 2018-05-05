import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import seq2seq_softmax_with_mask


class MultiplicativeAttentionWithMask(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiplicativeAttentionWithMask, self).__init__()
        self.encoder_linear = nn.Linear(input_size, output_size)
        self.decoder_linear = nn.Linear(input_size, output_size)

    def forward(self, decoder_hidden, encoder_hiddens, encoder_mask):
        """
        decoder_hidden: NxH
        encoder_hiddens: NxTxH
        """
        decoder_hidden_key = F.tanh(self.decoder_linear(decoder_hidden))
        encoder_hiddens_keys = F.tanh(self.encoder_linear(encoder_hiddens))
        weights = torch.bmm(encoder_hiddens_keys, decoder_hidden_key.unsqueeze(2))
        weights = seq2seq_softmax_with_mask(weights, encoder_mask)
        return torch.bmm(encoder_hiddens.transpose(1, 2), weights.unsqueeze(2))[:, :, 0]

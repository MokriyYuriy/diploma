import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTMDiscriminator(nn.Module):
    def __init__(self, alph, embedding_size, hidden_size):
        super(BiLSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.alph = alph
        self.embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=len(alph))
        self.forward_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(2 * hidden_size, 1)

    def reverse_sequence(self, input_sequence, mask):
        idx = torch.cumsum(mask, dim=1)
        idx = idx.max(1, keepdim=True)[0] - idx
        return torch.gather(input_sequence, 1, idx).detach()


    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        mask = self.alph.get_mask(input_sequence)
        reversed_input_sequence = self.reverse_sequence(input_sequence, mask)
        forward_embedding = self.embedding(input_sequence)
        backward_embedding = self.embedding(reversed_input_sequence)
        forward_out, _ = self.forward_lstm(forward_embedding)
        backward_out, _ = self.backward_lstm(backward_embedding)
        forward_hs = forward_out[range(batch_size), self.alph.get_length(input_sequence) - 1]
        backward_hs = backward_out[range(batch_size), self.alph.get_length(input_sequence) - 1]
        return self.output(torch.cat((forward_hs, backward_hs), dim=1))

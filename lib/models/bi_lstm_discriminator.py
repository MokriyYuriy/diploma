import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMDiscriminator(nn.Module):
    def __init__(self, alph, embedding_size, hidden_size):
        super(BiLSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.alph = alph
        self.embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=len(alph))
        self.bilstm = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, bidirectional=True, batch_first=True
        )
        self.output = nn.Linear(2 * hidden_size, 1)

    def forward(self, input_sequence):
        mask = self.alph.get_mask(input_sequence).unsqueeze(2)
        embedding = self.embedding(input_sequence) * mask
        out, _ = self.bilstm(embedding)
        forward_hs = out[range(out.size(0)), self.alph.get_length(input_sequence) - 1][:,:self.hidden_size]
        backward_hs = out[range(out.size(0)), [0] * (out.size(0))][:,self.hidden_size:]
        return F.sigmoid(
            self.output(torch.cat((forward_hs, backward_hs), dim=1))
        )

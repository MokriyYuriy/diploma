import torch.nn as nn
import torch.nn.functional as F


class LSTMDiscriminator(nn.Module):
    def __init__(self, alph, embedding_size, hidden_size):
        super(LSTMDiscriminator, self).__init__()
        self.alph = alph
        self.embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=len(alph))
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, input_sequence):
        embedding = self.embedding(input_sequence)
        out, _ = self.lstm(embedding)
        return F.sigmoid(
            self.output(out[range(out.size(0)), self.alph.get_length(input_sequence) - 1].view(out.size(0), -1)))

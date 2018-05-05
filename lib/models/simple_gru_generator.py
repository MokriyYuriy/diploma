import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .multiplicative_attention import MultiplicativeAttentionWithMask


class SimpleGRUEncoder(nn.Module):
    def __init__(self, alphabet, embedding_size, hidden_size):
        super(SimpleGRUEncoder, self).__init__()
        self.alphabet = alphabet
        self.embedding = nn.Embedding(num_embeddings=len(self.alphabet), embedding_dim=embedding_size)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input_sequence):
        embeddings = self.embedding(input_sequence)
        out, _ = self.gru(embeddings)
        return out, self.alphabet.get_mask(input_sequence)


class SimpleGRUDecoderWithAttention(nn.Module):
    def __init__(self, alphabet, embedding_size, hidden_size):
        super(SimpleGRUDecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.alphabet = alphabet
        self.embedding = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_size)
        self.gru_cell = nn.GRUCell(input_size=embedding_size + hidden_size, hidden_size=hidden_size)
        self.logit_linear = nn.Linear(hidden_size, len(alphabet))
        self.attention = MultiplicativeAttentionWithMask(hidden_size, embedding_size)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def forward(self, token, prev_h, encoder_hs, encoder_mask):
        embedding = self.embedding(token)
        attention = self.attention(prev_h, encoder_hs, encoder_mask)
        # print(attention.shape, embedding.shape)
        h = self.gru_cell(torch.cat((embedding, attention), dim=1), prev_h)
        out = self.logit_linear(h)
        return out, h


class SimpleGRUSupervisedSeq2Seq(nn.Module):
    def __init__(self, src_alphabet, dst_alphabet, embedding_size, hidden_size):
        super(SimpleGRUSupervisedSeq2Seq, self).__init__()
        self.encoder = SimpleGRUEncoder(src_alphabet, embedding_size, hidden_size)
        self.h_linear = nn.Linear(hidden_size, hidden_size)
        self.decoder = SimpleGRUDecoderWithAttention(dst_alphabet, embedding_size, hidden_size)

    def start(self, batch_size):
        return Variable(torch.from_numpy(np.repeat(self.decoder.alphabet.start_index, batch_size)))

    '''
    def middle_layer(self, out, mask):
        #print(mask.sum(1))
        return F.tanh(self.h_linear(out[range(out.shape[0]), mask.sum(1).long() - 1]))
    '''

    def forward(self, input_sequence, output_sequence):
        enc_out, enc_mask = self.encoder(input_sequence)
        dec_h = self.decoder.init_hidden(input_sequence.size(0))
        logits = []
        for x in output_sequence.transpose(0, 1)[:-1]:
            out, dec_h = self.decoder(x, dec_h, enc_out, enc_mask)
            logits.append(out)
        return F.log_softmax(torch.stack(logits, dim=1), dim=-1)

    def translate(self, word, strategy='', max_length=30, with_start_end=True):
        if isinstance(word, str):
            as_word = True
            input_sequence = Variable(torch.from_numpy(np.array([self.encoder.alphabet.letter2index(word)])))
        elif isinstance(word, torch.autograd.variable.Variable):
            as_word = False
            input_sequence = word
        else:
            assert False, "word argument must be str or numpy array"

        # print(input_sequence.shape)
        enc_out, enc_mask = self.encoder(input_sequence)
        hidden = self.decoder.init_hidden(input_sequence.size(0))
        tokens = self.start(input_sequence.size(0))
        # print(token.shape, hidden.shape)
        lst = [tokens]
        for i in range(max_length - 1):
            out, hidden = self.decoder(tokens, hidden, enc_out, enc_mask)
            tokens = out.max(1)[1]
            # print(token, out)
            lst.append(tokens)
            if as_word and tokens.data[0] == self.decoder.alphabet.end_index:
                break
        if as_word:
            return ''.join(model.decoder.alphabet.index2letter(
                [x.data[0] for x in lst],
                with_start_end=with_start_end)
            )
        else:
            return torch.stack(lst).transpose(0, 1)
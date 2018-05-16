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
    def __init__(self, alphabet, embedding_size, hidden_size, use_cuda=False):
        super(SimpleGRUDecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.alphabet = alphabet
        self.embedding = nn.Embedding(num_embeddings=len(alphabet), embedding_dim=embedding_size)
        self.gru_cell = nn.GRUCell(input_size=embedding_size + hidden_size, hidden_size=hidden_size)
        self.logit_linear = nn.Linear(hidden_size, len(alphabet))
        self.attention = MultiplicativeAttentionWithMask(hidden_size, embedding_size)
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return Variable(torch.cuda.FloatTensor(batch_size, self.hidden_size).fill_(0))
        else:
            return Variable(torch.zeros((batch_size, self.hidden_size)))

    def forward(self, token, prev_h, encoder_hs, encoder_mask):
        embedding = self.embedding(token)
        attention = self.attention(prev_h, encoder_hs, encoder_mask)
        # print(attention.shape, embedding.shape)
        h = self.gru_cell(torch.cat((embedding, attention), dim=1), prev_h)
        out = self.logit_linear(h)
        return out, h


class SimpleGRUSupervisedSeq2Seq(nn.Module):
    GREEDY = 'greedy'
    SAMPLING = 'sampling'

    def __init__(self, src_alphabet, dst_alphabet, embedding_size, hidden_size, use_cuda=False):
        super(SimpleGRUSupervisedSeq2Seq, self).__init__()
        self.encoder = SimpleGRUEncoder(src_alphabet, embedding_size, hidden_size)
        self.h_linear = nn.Linear(hidden_size, hidden_size)
        self.decoder = SimpleGRUDecoderWithAttention(dst_alphabet, embedding_size, hidden_size, use_cuda)
        self.use_cuda = use_cuda

    def start(self, batch_size):
        if self.use_cuda:
            return Variable(torch.cuda.LongTensor(batch_size).fill_(self.decoder.alphabet.start_index))
        else:
            return Variable(torch.LongTensor(batch_size).fill_(self.decoder.alphabet.start_index))

    def end(self, batch_size):
        if self.use_cuda:
            return Variable(torch.cuda.LongTensor(batch_size).fill_(self.decoder.alphabet.end_index))
        else:
            return Variable(torch.LongTensor(batch_size).fill_(self.decoder.alphabet.end_index))

    def end_mask(self, batch_size):
        if self.use_cuda:
            return Variable(torch.cuda.ByteTensor(batch_size).fill_(0))
        else:
            return Variable(torch.ByteTensor(batch_size).fill_(0))

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

    def translate(self, words, strategy=GREEDY, return_logits=False, max_length=30, with_start_end=True):
        if isinstance(words, str):
            as_word = True
            input_sequence = torch.from_numpy(np.array([self.encoder.alphabet.letter2index(words)]))
            if self.use_cuda:
                input_sequence = input_sequence.cuda()
            input_sequence = Variable(input_sequence)
        elif isinstance(words, torch.autograd.variable.Variable):
            as_word = False
            input_sequence = words
        else:
            assert False, "word argument must be str or numpy array"

        # print(input_sequence.shape)
        enc_out, enc_mask = self.encoder(input_sequence)
        hidden = self.decoder.init_hidden(input_sequence.size(0))
        tokens = self.start(input_sequence.size(0))
        end = self.end(input_sequence.size(0))
        end_mask = self.end_mask(input_sequence.size(0))
        # print(token.shape, hidden.shape)
        lst = [tokens]
        logits = []
        for i in range(max_length - 1):
            out, hidden = self.decoder(tokens, hidden, enc_out, enc_mask)
            if strategy == self.GREEDY:
                tokens = out.max(1)[1]
            elif strategy == self.SAMPLING:
                tokens = torch.multinomial(F.log_softmax(out), 1)
            else:
                assert False, "provided value of strategy param is not appropriate"
            if return_logits:
                logits.append(F.log_softmax(out))
            # print(token, out)
            end_mask |= (tokens == end)
            tokens = tokens.masked_scatter_(end_mask, end)
            lst.append(tokens)
            if as_word and tokens.data[0] == self.decoder.alphabet.end_index:
                break
        if as_word:
            return ''.join(self.decoder.alphabet.index2letter(
                [x.data[0] for x in lst],
                with_start_end=with_start_end)
            )
        else:
            result_sequence = torch.stack(lst).transpose(0, 1)
            if return_logits:
                return result_sequence, torch.stack(logits).transpose(0, 1)
            return result_sequence
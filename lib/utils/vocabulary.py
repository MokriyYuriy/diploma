import torch as torch
from torch.autograd import Variable


class Alphabet:
    MAX_LENGTH = 30
    START = '__START__'
    END = '_END_'

    def __init__(self, max_length=MAX_LENGTH):
        """Initialize the class which works with letter and index representations of sequences.
        Parameters
        ----------
        max_length : int
            The largest permitted length for sequence. Longer sequences are cropped.
        """
        self.max_length = max_length
        self.letter2index_ = {Alphabet.START: 0, Alphabet.END: 1}
        self.index2letter_ = [Alphabet.START, Alphabet.END]

    def get_index(self, letter):
        if letter not in self.letter2index_:
            self.letter2index_[letter] = len(self.index2letter_)
            self.index2letter_.append(letter)
        return self.letter2index_[letter]

    @property
    def start_index(self):
        return self.letter2index_[Alphabet.START]

    @property
    def end_index(self):
        return self.letter2index_[Alphabet.END]

    def index2letter(self, x, with_start_end=True):
        result = []
        for index in x[0 if with_start_end else 1:]:
            if index == self.end_index:
                if with_start_end:
                    result.append(self.index2letter_[index])
                break
            result.append(self.index2letter_[index])
        return ''.join(result)

    def letter2index(self, word):
        lst = [self.get_index(letter) for letter in word]
        return [self.start_index] + lst[:self.max_length - 2] + [self.end_index] * max(1,
                                                                                       self.max_length - len(lst) - 1)

    def __len__(self):
        return len(self.index2letter_)

    # torch utils
    def get_length(self, input_sequence):
        """Infers the lengths of the sequences in batch

        input_sequence: Tensor NxT

        returs: Tensor N
        """
        return (input_sequence == self.end_index).max(dim=1)[1] + 1

    def get_mask(self, input_sequence):
        """Infers the mask of the sequences in batch

        input_sequence: Tensor NxT

        returns: Tensor NxT contained 0s and 1s.
        """
        return (torch.cumsum(input_sequence == self.end_index, dim=1) < 2).float()

    def get_one_hot_repr(self, input_sequence):
        """Produces one_hot representation from label representation/

        input_sequence: LongTensor NxT

        returns: FloatTensor NxTxH
        """

        onehot = torch.FloatTensor(*input_sequence.shape, len(self)).zero_()
        onehot.scatter_(2, input_sequence.unsqueeze(2), 1.)

        return onehot

    def get_mask_for_3D_array(self, input_sequence, input_array):
        length_mask = self.get_mask(input_sequence)
        input_sequence_mask = torch.zeros_like(input_array).scatter_(
            2, input_sequence.view(*input_sequence.shape, 1).contiguous(), 1.0)
        return input_sequence_mask * length_mask.view(*length_mask.shape, 1)

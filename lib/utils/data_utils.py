import numpy as np


def load_pair_dataset(filename, alph1, alph2):
    x, y = [], []
    with open(filename, 'r') as ftr:
        for line in ftr:
            try:
                word1, word2 = line.strip().split('\t')
            except ValueError:
                continue
            x.append(alph1.letter2index(word1))
            y.append(alph2.letter2index(word2))
    return np.array(x), np.array(y)

def split_file_into_two(filename, train_filename, test_filename, test_size=0.1, random_seed=179):
    with open(filename, 'r') as ftr, open(train_filename, 'w') as train_ftw, \
        open(test_filename, 'w') as test_ftw:
        np.random.seed(random_seed)
        for line in ftr:
            if np.random.rand() < test_size:
                test_ftw.write(line)
            else:
                train_ftw.write(line)

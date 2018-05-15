#import editdistance as ed
import nltk.translate.bleu_score as bl
import numpy as np
#from tqdm import tqdm_notebook


def compute_bleu_score(model, src_words, trg_words):
    return _compute_metric_average(model, src_words, trg_words, lambda x, y: bl.sentence_bleu([list(x)], list(y)))

def compute_editdistance(model, src_words, trg_words):
    return _compute_metric_average(model, src_words, trg_words, ed.eval)

def compute_accuracy(model, src_words, trg_words):
    return _compute_metric_average(model, src_words, trg_words, lambda x, y: x == y)

def _compute_metric_average(model, src_words, trg_words, metric):
    scs = [metric(model.translate(x, with_start_end=False), y) for x, y in zip(src_words, trg_words)]
    return np.mean(scs)
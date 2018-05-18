#import editdistance as ed
import nltk.translate.bleu_score as bl
import numpy as np
#from tqdm import tqdm_notebook


def compute_corpus_bleu_score(model, src_words, trg_words, batch_size=100):
    translations = []
    for i in range(0, len(src_words), batch_size):
        translations.extend(model.translate(src_words[i:i+batch_size], with_start_end=False))
    translations = [[list(translation)] for translation in translations]
    ground_truth = [list(word) for word in trg_words]
    return bl.corpus_bleu(translations, ground_truth)

def compute_cycle_corpus_bleu_score(forward_model, reversed_model, words, batch_size=100):
    translations = []
    for i in range(0, len(words), batch_size):
        translations.extend(
            reversed_model.translate(
                forward_model.translate(words[i:i+batch_size], with_start_end=False),
                with_start_end=False
            )
        )
    translations = [[list(translation)] for translation in translations]
    ground_truth = [list(word) for word in words]
    return bl.corpus_bleu(translations, ground_truth)

def compute_sentence_bleu_score(model, src_words, trg_words):
    return _compute_metric_average(model, src_words, trg_words, lambda x, y: bl.sentence_bleu([list(x)], list(y)))

def compute_editdistance(model, src_words, trg_words):
    return _compute_metric_average(model, src_words, trg_words, ed.eval)

def compute_accuracy(model, src_words, trg_words):
    return _compute_metric_average(model, src_words, trg_words, lambda x, y: x == y)

def _compute_metric_average(model, src_words, trg_words, metric):
    scs = [metric(model.translate(x, with_start_end=False), y) for x, y in zip(src_words, trg_words)]
    return np.mean(scs)
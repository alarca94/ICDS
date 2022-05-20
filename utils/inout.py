import json
import os
import pandas as pd

from gensim.models import KeyedVectors

from utils.functions import get_comparison_words, update_dict

DATA_PATH = './DATA_SETS'
DATASETS = ['MEN', 'SICK', 'STS']
DATA_PATH2 = './data'
WV_FILE = 'GoogleNews-vectors-negative300.bin'


def read_data():
    # Read the data
    experiments_cols = ['text1', 'text2', 'gs']
    sentences = {}
    experiments = {}

    for dataset in DATASETS:
        files = os.listdir(os.path.join(DATA_PATH, dataset))
        sentences_file = [f for f in files if f.endswith('SENTENCES2.pkl')][0]
        experiments_file = [f for f in files if f.endswith('SIMILARITIES.txt')][0]

        sentences[dataset] = pd.read_pickle(os.path.join(DATA_PATH, dataset, sentences_file))
        experiments[dataset] = pd.read_csv(os.path.join(DATA_PATH, dataset, experiments_file), names=experiments_cols,
                                           header=None, index_col=0, sep='\t')

    # Read the stopwords file
    with open(os.path.join(DATA_PATH, 'stopwords_en.txt'), 'r', encoding='utf-8') as f:
        stopwords = [l.rstrip() for l in f.readlines()]

    return sentences, experiments, stopwords


def read_word2vec():
    return KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH2, WV_FILE), binary=True)


def read_stopwords():
    with open(os.path.join(DATA_PATH, 'stopwords_en.txt'), 'r', encoding='utf-8') as f:
        stopwords = [line.rstrip() for line in f.readlines()]
    return stopwords


def read_comparison_words():
    with open(os.path.join(DATA_PATH, 'comparison_words.txt'), 'r', encoding='utf-8') as f:
        comp_ws = [line.rstrip() for line in f.readlines()]
    return comp_ws


def save_comparison_words():
    sentences, experiments, _ = read_data()
    word2vec = read_word2vec()
    comp_words = get_comparison_words(sentences['MEN'], word2vec, read_stopwords())
    with open(os.path.join(DATA_PATH, 'comparison_words.txt'), 'w') as f:
        f.write('\n'.join(comp_words))


def write_results(results, filename):
    prev_results = results.copy()
    if os.path.isfile(os.path.join(DATA_PATH, 'results', filename)):
        with open(os.path.join(DATA_PATH, 'results', filename), 'r') as fp:
            prev_results = json.load(fp)
        update_dict(prev_results, results)

    with open(os.path.join(DATA_PATH, 'results', filename), 'w') as fp:
        json.dump(prev_results, fp)

import os
import re
import nltk
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from utils.config_params import RANDOM_SEED

K = 1
DATA_PATH = './DATA_SETS'


def icm(v1, v2, c=1, c1=1, c2=1):
    """
    Compute the similarity between two vectors considering their information quantity
    If c is an array, its shape must be (n_comp, 1)
    v1 and v2 shapes are (n_comp, vec_dim)
    """
    if type(c) == list:
        return (c1-c) * np.broadcast_to(_iq(v1).reshape(-1, 1), (v1.shape[0], v2.shape[0])) + \
               (c2-c) * np.broadcast_to(_iq(v2).reshape(1, -1), (v1.shape[0], v2.shape[0])) + \
               c * np.dot(v1, v2.T)
    else:
        return (c1-c) * _iq(v1).reshape(-1, 1) + (c2-c) * _iq(v2).reshape(1, -1) + c * np.dot(v1, v2.T)
    
    
def dot_product(v1, v2):
    return np.dot(v1, v2.T)


def euclidean_score(v1, v2):
    return -euclidean_distances(v1, v2)


def _iq(vec):
    """
    Compute the information quantity of a word vector
    """
    return (vec.T ** 2).sum(axis=0)


def _iq2(vec, K=1):
    """
    Compute the information quantity of a word vector with the estimate adjustment
    """
    return np.sum(vec.T ** 2, axis=0) + np.log(K)


def _sem_ori_comp(vec1, vec2, gamma):
    """
    Compute the semantic orientation of the _composition function
    """
    return (gamma * vec1 + vec2) / np.linalg.norm(gamma * vec1 + vec2)


def _mag_comp(vec1, vec2, alpha, beta):
    """
    Compute the magnitude of the _composition function
    """
    return np.sqrt((alpha-beta) * (_iq(vec1) + _iq(vec2)) + beta * np.dot(vec1, vec2))


def _mag_comp2(vec1, vec2, alpha, beta, K=1):
    """
    Compute the magnitude of the _composition function (version 2)
    """
    return np.sqrt(max(alpha * (_iq(vec1) + _iq(vec2)) - beta * np.dot(vec1, vec2) + (2*alpha+beta) * np.log(K), 0.0))


def _mag_comp3(vec1, vec2, alpha=1):
    """
    Compute the magnitude of the _composition function (version 3)
    K no longer affects the composition
    Alpha is defaulted to 1 as per theoretical constraints
    Beta is computed as |Y|/|X| with Y being the vector with lower magnitude
    """
    iqs = sorted([_iq(vec1), _iq(vec2)])
    # beta = np.sqrt(iqs[0] / iqs[1])
    # beta = iqs[0] / iqs[1]
    beta = iqs[0] ** 2 / (iqs[0] * iqs[1])
    return np.sqrt(alpha * np.sum(iqs) - beta * np.dot(vec1, vec2))


def _comp(vec1, vec2, alpha, beta, gamma, K=1):
    """
    Compose the vectors of two words
    """
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    
    if beta is None:
        return _sem_ori_comp(vec1, vec2, gamma) * _mag_comp3(vec1, vec2, alpha)

    return _sem_ori_comp(vec1, vec2, gamma) * _mag_comp2(vec1, vec2, alpha, beta, K)


def comp_LM(v1g, v2g, v1l, v2l, alpha, beta, gamma, K=1):
    """
    v1g: First token general context embedding
    v2g: Second token general context embedding
    v1l: First token local context embedding
    v2l: Second token local context embedding
    """
    if beta is None:
        return _sem_ori_comp(v1l, v2l, gamma) * _mag_comp3(v1g, v2g, alpha)

    return _sem_ori_comp(v1l, v2l, gamma) * _mag_comp2(v1g, v2g, alpha, beta, K)


def _sum(vec1, vec2, alpha=None, beta=None, gamma=None, K=None):
    """
    Compose the vectors of two words by summing the vectors
    """
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1

    return vec1 + vec2


def _average(vec1, vec2, alpha=None, beta=None, gamma=None, K=None):
    """
    Compose the vectors of two words by summing the vectors
    """
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1

    return (vec1 + vec2) / 2


def _total_avg(elements, word2vecs, stopwords=[]):
    vecs = []
    elements = flatten(elements)
    for element in elements:
        if element not in stopwords and element in word2vecs.vocab.keys():
            vecs.append(word2vecs[element])
    if vecs:
        vecs = np.array(vecs)
        return np.mean(vecs, axis=0)
    return np.zeros((len(word2vecs[list(word2vecs.vocab.keys())[0]]),))


def _total_sum(elements, word2vecs, stopwords=[]):
    vecs = []
    elements = flatten(elements)
    for element in elements:
        if element not in stopwords and element in word2vecs.vocab.keys():
            vecs.append(word2vecs[element])
    if vecs:
        vecs = np.array(vecs)
        return np.sum(vecs, axis=0)
    return np.zeros((len(word2vecs[list(word2vecs.vocab.keys())[0]]),))


def compose(elements, word2vecs, alpha, beta, gamma, K=15, start='left', comp_func=_comp, stopwords=[]):
    init_id = 0
    composition = None

    ids = list(range(len(elements)))
    if start == 'right':
        ids = ids[::-1]

    # Assign to the composition the vector of the first component of the sentence (compute composition if necessary)
    while composition is None and init_id < len(ids):
        if isinstance(elements[ids[init_id]], list):
            composition = compose(elements[ids[init_id]], word2vecs, alpha, beta, gamma, K, start, comp_func, stopwords)
        else:
            # Check if the current word exists in the dictionary
            if elements[ids[init_id]] in word2vecs.vocab.keys() and elements[ids[init_id]].lower() not in stopwords:
                composition = word2vecs[elements[ids[init_id]]]
        init_id += 1

    # If there are still words to consider, continue composing the sentence
    if init_id < len(ids):
        for i in ids[init_id:]:
            if isinstance(elements[i], list):
                v1 = compose(elements[i], word2vecs, alpha, beta, gamma, K, start, comp_func, stopwords)
                composition = comp_func(composition, v1, alpha, beta, gamma, K)
            else:
                if elements[i] in word2vecs.vocab.keys() and elements[i].lower() not in stopwords:
                    v1 = word2vecs[elements[i]]
                    composition = comp_func(composition, v1, alpha, beta, gamma, K)
    
    return composition


def sort_def(definition, criteria='seq', parser=None, start='left'):
    """
    Given a definition of a word, sort it according to a specific criteria
    # seq = Sequential
    # syn = Syntactic
    # dep = Dependency
    """
    if criteria == 'seq':
        return list(parser.tokenize(definition))
    elif criteria == 'syn':
        return _sort_syntax(definition, parser)
    elif criteria == 'dep':
        return _sort_dep(definition, parser, start)
    else:
        return definition


def _sort_syntax(sent, parser):
    """
    Given a raw sentence and a constituents parser, extract the sorted sentence for _composition
    """
    tree = parser.raw_parse(sent, keepPunct=True)
    t = next(tree)

    leaves = t.leaves()
    columns = {i: l for i, l in enumerate(leaves)}
    max_depth = t.height() - 1
    leaves_pos = []
    for i in range(len(t.leaves())):
        leaves_pos.append(t.leaf_treeposition(i))

    leaves_pos = pd.DataFrame(np.array([list(pos) + [np.nan] * (max_depth - len(pos)) for pos in leaves_pos]).T,
                              columns=columns.keys())

    return _get_struct_syn(leaves_pos, leaves_pos.columns, columns, 0)


def _sort_dep(sent, dep_parser, start):
    """
    Given a raw sentence and a dependency parser, extract the sorted sentence for _composition
    """
    tokens = pd.Series(dep_parser.tokenize(sent))
    skip_tokens = list('()"`.[]—{}') + ["''", '’’', '5 1/2', '1 1/2','‘‘']
    
    p = list(dep_parser.raw_parse(sent, keepPunct=True))[0]
    t_tokens = [node['word'] for key, node in p.nodes.items()]
    miss_tokens = tokens[~tokens.isin(t_tokens) & ~tokens.isin(skip_tokens)].tolist()
    
    if miss_tokens:
        print('Adding missed tokens: ' + str(miss_tokens) + ' /// ' + sent)
        if start == 'left':
            return [_get_struct_dep(p, start)] + miss_tokens
        else:
            return miss_tokens + [_get_struct_dep(p, start)]
    
    return _get_struct_dep(p, start)


def _get_struct_syn(l_pos, cols_ixs, cols_map, depth):
    struct = []
    if len(cols_ixs) > 1:
        sub_data = l_pos.loc[depth, cols_ixs]
        groups = sub_data.groupby(sub_data).apply(lambda s: s.index.tolist()).tolist()
        if len(groups) == 1:
            struct = _get_struct_syn(l_pos, groups[0], cols_map, depth + 1)
        else:
            for group in groups:
                struct.append(_get_struct_syn(l_pos, group, cols_map, depth + 1))

        return struct
    else:
        return cols_map[cols_ixs[0]]


def _get_dependencies(p, gov):
    dep = {}
    for key, node in p.nodes.items():
        if node['head'] == gov:
            dep[node['address']] = _get_dependencies(p, key)
    return dep


def _transform_dep_old(dep, p):
    # ['a', ['native', ['or', 'inhabitant', 'States', ['of', 'the', 'United']]]]
    struct = []
    for key, val in dep.items():
        struct.append(p.nodes[key]['word'])
        if val:
            struct.append(_transform_dep_old(val, p))

    return struct


def _transform_dep(dep, p, start):
    # ['a', ['native', 'or', 'inhabitant', ['States', 'of', 'the', 'United']]]
    struct = []
    for key, val in dep.items():
        if val:
            if start == 'left':
                struct.append([p.nodes[key]['word']] + _transform_dep(val, p, start))
            elif start == 'right':
                struct.append(_transform_dep(val, p, start) + [p.nodes[key]['word']])
        else:
            struct.append(p.nodes[key]['word'])

    return struct


def _get_struct_dep(p, start):
    dep = _get_dependencies(p, 0)
    struct = _transform_dep(dep, p, start)[0]
    return struct


def experiments(test):
    if test == 1:
        orders = ['left', 'right']
        criteria = ['seq', 'syn', 'dep']
        alphas = np.array([3])
        betas = alphas - 1
        gammas = np.ones((len(betas),))
        cs = np.array([1.2])
        c1s = np.ones((len(cs),))
        c2s = np.ones((len(cs),))
    elif test == 2:
        orders = ['right']
        criteria = ['seq', 'syn', 'dep']
        alphas = np.round(np.arange(0, 4, 0.2), 1)
        betas = np.round(alphas - 1, 1)
        gammas = np.ones((len(betas),))
        cs = np.array([1.2])
        c1s = np.ones((len(cs),))
        c2s = np.ones((len(cs),))
    elif test == 3:
        orders = ['right']
        criteria = ['seq', 'syn', 'dep']
        alphas = np.array([3])
        betas = alphas - 1
        gammas = np.ones((len(betas),))
        cs = np.round(np.arange(1, 2.10, 0.01), 1)
        c1s = np.ones((len(cs),))
        c2s = np.ones((len(cs),))

    for order in orders:
        for cri in criteria:
            for alpha, beta, gamma in zip(alphas, betas, gammas):
                for c, c1, c2 in zip(cs, c1s, c2s):
                    yield order, cri, alpha, beta, gamma, c, c1, c2

                    
def extract_comps_from_words_2(words, word_defs, order, word_vectors, alpha, beta, gamma):
    all_comps = []
    for word in words:
        comps = np.array([])
        for criteria in ['seq', 'syn', 'dep']:
            # Compose the definition
            composition = compose(word_defs[word][criteria], word_vectors, alpha=alpha, beta=beta, 
                                  gamma=gamma, start=order)
            comps = np.concatenate((comps, composition))
        all_comps.append(comps)
    return all_comps


def samples_gen_2(words, word_defs, batch_size, order, word_vectors, alpha, beta, gamma):
    batch_ix = 0
    
    while True:
        while (batch_ix + 1) * batch_size < len(words):
            batch_words = words[batch_ix * batch_size:(batch_ix + 1) * batch_size]

            # For each word in the batch, compute the compositionality
            x_train = extract_comps_from_words_2(batch_words, word_defs, order, word_vectors, 
                                                 alpha, beta, gamma)

            y_train = [word_vectors[w] for w in batch_words]

            yield np.array(x_train), np.array(y_train)

            batch_ix +=1

        x_train = extract_comps_from_words_2(words[batch_ix * batch_size:], word_defs, order, 
                                             word_vectors, alpha, beta, gamma)

        y_train = [word_vectors[w] for w in words[batch_ix * batch_size:]]

        yield np.array(x_train), np.array(y_train)

        batch_ix = 0
        
        
def flatten(lista):
    if type(lista) == str:
        return [lista]
    else:
        result = []
        for l in lista: 
            result.extend(flatten(l))
        return result

        
def aligned_similarity(s1, s2, word2vecs):
    s1 = flatten(s1)
    s2 = flatten(s2)
    
    s1_m = np.array([word2vecs[w] for w in s1 if w in word2vecs.vocab.keys()])
    s2_m = np.array([word2vecs[w] for w in s2 if w in word2vecs.vocab.keys()])
    
    if s1_m.size == 0 or s2_m.size == 0:
        return 0

    sims = cosine_similarity(s1_m, s2_m)

    align_sims = np.max(sims, axis=(0 if s2_m.shape[0] > s1_m.shape[0] else 1))

    return np.mean(align_sims)


def to_lowercase(elements):
    if isinstance(elements, list):
        return [to_lowercase(e) for e in elements]
    else:
        return elements.lower()


def dict2table(dic):
    table = []
    for key, val in dic.items():
        if type(val) == dict:
            table.extend([[key] + row for row in dict2table(val)])
        else:
            table.append([key, val])
    return table


def get_results(metrics, sw_modes, suffix=''):
    columns = ['similarity', 'stopwords', 'dataset', 'direction', 'struct', 'conf', 'score']
    fset_table = pd.DataFrame(columns=columns)
    sset_table = pd.DataFrame(columns=columns)
    for metric in metrics:
        for sw_mode in sw_modes:
            filename = f'{sw_mode}_{metric}_'
            main_results_1 = read_results(f'final_{filename}{suffix}1.json')
            sts_results = read_results(f'finalSTS_{filename}{suffix}1.json')
            main_results_1.update(sts_results)
            main_results_2 = read_results(f'final_{filename}2.json')
            fset_table = pd.concat([fset_table, normalize_results(main_results_1, metric, sw_modes[sw_mode])],
                                   ignore_index=True)
            sset_table = pd.concat([sset_table, normalize_results(main_results_2, metric, sw_modes[sw_mode])],
                                   ignore_index=True)

    fset_table['score'] = fset_table['score'].astype(np.float64)
    sset_table['score'] = sset_table['score'].astype(np.float64)
    return fset_table, sset_table


def read_results(results_file):
    results_path = './DATA_SETS/results'

    with open(os.path.join(results_path, results_file), 'r') as f:
        results = json.load(f)

    return results


def normalize_results(results, metric, sw_mode):
    table = dict2table(results)
    sum_table = pd.DataFrame([row for row in table if 'SUM' in row and {'BERT', 'GPT', 'S-BERT'}.isdisjoint(set(row))],
                             columns=['dataset', 'f_x', 'score'])
    sbert_table = pd.DataFrame([row for row in table if 'S-BERT' in row and {'BERT', 'GPT'}.isdisjoint(set(row))],
                                columns=['dataset', 'conf', 'score'])
    lm_table = pd.DataFrame([row for row in table if 'BERT' in row or 'GPT' in row],
                              columns=['dataset', 'direction', 'conf', 'f_x', 'score'])
    reg_table = pd.DataFrame([row for row in table if {'SUM', 'BERT', 'GPT', 'S-BERT'}.isdisjoint(set(row))],
                             columns=['dataset', 'direction', 'struct', 'f_x', 'score'])
    # Add missing columns to separate tables
    sum_table['direction'] = [None] * sum_table.shape[0]
    sum_table['struct'] = ['SEQ'] * sum_table.shape[0]
    sum_table['conf'] = ['W2V'] * sum_table.shape[0]
    reg_table['conf'] = ['W2V'] * reg_table.shape[0]
    sbert_table['direction'] = [None] * sbert_table.shape[0]
    sbert_table['struct'] = ['SEQ'] * sbert_table.shape[0]
    sbert_table['f_x'] = ['GLOBAL_AVG'] * sbert_table.shape[0]
    lm_table['struct'] = ['SEQ'] * lm_table.shape[0]

    # Beautify and correct some strings
    reg_table.struct = reg_table.struct.str.upper()
    reg_table = reg_table.replace({'direction': {'left': 'L2R', 'right': 'R2L'}})
    lm_table = lm_table.replace({'direction': {'left': 'L2R', 'right': 'R2L'}})
    lm_table.loc[lm_table.f_x == 'CLS', 'conf'] = 'BERT_CLS'

    # Join the tables and add general columns
    table = pd.concat([reg_table, sum_table, lm_table, sbert_table], ignore_index=True)
    table['similarity'] = [metric.upper()] * table.shape[0]
    table['stopwords'] = [sw_mode] * table.shape[0]

    return table


def get_spans(width, n_bars):
    if n_bars % 2:
        spans = [i * width for i in range(-n_bars // 2 + 1, n_bars // 2 + 1)]
    else:
        spans = [(i + 0.5) * width for i in range(-n_bars // 2, n_bars // 2)]
    return spans


def get_rects(table, col_table, ax, options, opt_mapper, cs, hs, width, spans, x):
    rects = []
    for i, opt in enumerate(options):
        rects.append(ax.bar(x + spans[i],
                            table[table[col_table] == opt].score.values,
                            width,
                            label=opt_mapper.get(opt, opt),
                            color=cs[i],
                            hatch=hs[i],
                            edgecolor='k'))
    return rects


def plot_comp_dataset_lm(table, title_main=None, y_label='SPEARMAN', image_name=None, dataset_label=None,
                         legend_loc='upper left'):
    """
    Horizontal axis: Datasets
    Vertical axis: Score for best configuration
    Series: Composition function
    """
    cos_table = table[(table.similarity == 'COS') & (table.struct == 'SEQ') & (table.dataset != 'STS')]

    data_sorter = ['MSRpar', 'MSRvid', 'answer-answer', 'images', 'track5', 'SICK', 'MEN']  # 'STS'
    conf_sorter = ['GPT', 'BERT', 'BERT_CLS', 'S-BERT', 'W2V']
    data_sorter = [v for v in data_sorter if v in table.dataset.unique()]
    conf_sorter = [v for v in conf_sorter if v in table.conf.unique()]
    cos_table.dataset = cos_table.dataset.astype('category')
    cos_table.dataset.cat.set_categories(data_sorter, inplace=True)
    cos_table.conf = cos_table.conf.astype('category')
    cos_table.conf.cat.set_categories(conf_sorter, inplace=True)

    cos_table = cos_table.loc[cos_table.groupby(['dataset', 'conf'])['score'].idxmax()]
    cos_table = cos_table.sort_values(['dataset', 'conf'])

    # conf_map = {'AVG': 'AVG   (\u03b1=1/4, \u03b2=1/2)',
    #             'IND': 'IND    (\u03b1=1, \u03b2=0)',
    #             'JOINT': 'JOINT  (\u03b1=1, \u03b2=1)',
    #             'INF': 'INF     (\u03b1=1, \u03b2\u2192opt.)',
    #             'SUM': 'SUM   (\u03b1=1, \u03b2=-2)'}
    data_map = {'MEN': 'DEF2DEF',
                'MSRpar': 'STS-MSRpar',
                'MSRvid': 'STS-MSRvid',
                'answer-answer': 'STS-answer-answer',
                'images': 'STS-images',
                'track5': 'STS-track5'}
    conf_map = {'GPT': 'GPT2-L2 + SEQ + BEST F_X + COS',
                'BERT': 'BERT + SEQ + BEST F_X + COS',
                'BERT_CLS': 'BERT (CLS) + COS',
                'S-BERT': 'S-BERT + COS',
                'W2V': 'W2V + SEQ + BEST F_X + COS'}

    n_confs = cos_table.conf.nunique()
    n_datasets = cos_table.dataset.nunique()
    configs = cos_table.conf.unique()
    datasets = cos_table.dataset.unique()

    x = np.arange(n_datasets)  # the label locations
    width = 1 / (n_confs + 1)  # the width of the bars

    fig, ax = plt.subplots(figsize=[12, 5])
    # cs = ['#FFFFFF'] * 5
    # cs = ['#1E245E', '#9DFEF7',  # '#E121EE', '#890192',
    #       '#0294E3', '#3702D1', '#23D207']
    cs = ['#AAFDE9', '#AAE3FF', '#AAE3FD', '#AAACFD', '#F9FDAA']
    hs = ['///', '\\\\', '++', '*', '..']

    spans = get_spans(width, n_confs)
    rects = get_rects(cos_table, 'conf', ax, configs, conf_map, cs, hs, width, spans, x)

    ax.set_ylabel(y_label)
    if title_main is None:
        title_main = 'Sentence-Sentence similarity prediction'
    # ax.set_title(f'{title_main} by composition across datasets')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    if dataset_label is None:
        rotation = 0
        if y_label == 'SPEARMAN':
            rotation = 45
        ax.set_xticklabels([data_map.get(d, d) for d in datasets], rotation=rotation)
    else:
        ax.set_xticklabels([dataset_label])
    # ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    ax.legend(loc=legend_loc)

    # for i, rect in enumerate(rects):
    #     autolabel(cos_table[cos_table.conf==config].direction.values, rect)

    fig.tight_layout()
    plt.grid()
    if image_name is not None:
        plt.savefig(os.path.join(DATA_PATH, 'images', image_name),
                    bbox_inches='tight')
    else:
        plt.show()


def plot_comp_dataset_w2v(table, title_main=None, y_label='SPEARMAN', image_name=None, dataset_label=None,
                          legend_loc='upper left'):
    """
    Horizontal axis: Datasets
    Vertical axis: Score for best configuration
    Series: Composition function
    """
    cos_table = table[(table.similarity == 'COS') & (table.conf == 'W2V') & (table.dataset != 'STS')]

    data_sorter = ['MSRpar', 'MSRvid', 'answer-answer', 'images', 'track5', 'SICK', 'MEN']  # 'STS'
    fx_sorter = ['AVG', 'SUM', 'IND', 'JOINT', 'INF']
    data_sorter = [v for v in data_sorter if v in table.dataset.unique()]
    fx_sorter = [v for v in fx_sorter if v in table.f_x.unique()]
    cos_table.dataset = cos_table.dataset.astype('category')
    cos_table.dataset.cat.set_categories(data_sorter, inplace=True)
    cos_table.f_x = cos_table.f_x.astype('category')
    cos_table.f_x.cat.set_categories(fx_sorter, inplace=True)

    cos_table = cos_table.loc[cos_table.groupby(['dataset', 'f_x'])['score'].idxmax()]
    cos_table = cos_table.sort_values(['dataset', 'f_x'])

    # conf_map = {'AVG': 'AVG   (\u03b1=1/4, \u03b2=1/2)',
    #             'IND': 'IND    (\u03b1=1, \u03b2=0)',
    #             'JOINT': 'JOINT  (\u03b1=1, \u03b2=1)',
    #             'INF': 'INF     (\u03b1=1, \u03b2\u2192opt.)',
    #             'SUM': 'SUM   (\u03b1=1, \u03b2=-2)'}
    data_map = {'MEN': 'DEF2DEF',
                'MSRpar': 'STS-MSRpar',
                'MSRvid': 'STS-MSRvid',
                'answer-answer': 'STS-answer-answer',
                'images': 'STS-images',
                'track5': 'STS-track5'}
    fx_map = {fx: f'W2V + BEST STRUCTURE + F_{fx} + COS' for fx in fx_sorter}

    n_fxs = cos_table.f_x.nunique()
    n_datasets = cos_table.dataset.nunique()
    fxs = cos_table.f_x.unique()
    datasets = cos_table.dataset.unique()

    x = np.arange(n_datasets)  # the label locations
    width = 1 / (n_fxs + 1)  # the width of the bars

    fig, ax = plt.subplots(figsize=[12, 5])
    # cs = ['#FFFFFF'] * 5
    # cs = ['#FFAF74', '#F3FF00', '#A0FFA1',  # '#E121EE', '#890192',
    #       '#00EA04', '#004B02']
    cs = ['#FDE3A1', '#F3FF00', '#AAFDE3', '#AAFDB5', '#D1FDAA']
    hs = ['///', '\\\\', '++', '*', '..']

    spans = get_spans(width, n_fxs)
    rects = get_rects(cos_table, 'f_x', ax, fxs, fx_map, cs, hs, width, spans, x)

    ax.set_ylabel(y_label)
    if title_main is None:
        title_main = 'Sentence-Sentence similarity prediction'
    # ax.set_title(f'{title_main} by composition across datasets')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    if dataset_label is None:
        rotation = 0
        if y_label == 'SPEARMAN':
            rotation = 45
        ax.set_xticklabels([data_map.get(d, d) for d in datasets], rotation=rotation)
    else:
        ax.set_xticklabels([dataset_label])
    # ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    ax.legend(loc=legend_loc)

    # for i, rect in enumerate(rects):
    #     autolabel(cos_table[cos_table.conf==config].direction.values, rect)

    fig.tight_layout()
    plt.grid()
    if image_name is not None:
        plt.savefig(os.path.join(DATA_PATH, 'images', image_name),
                    bbox_inches='tight')
    else:
        plt.show()


def plot_similarity_composition(table, title_main=None, y_label='SpearmanR', image_name=None, legend_loc='upper left'):
    """
    Horizontal axis: Composition functions
    Vertical axis: Mean Pearson across datasets per composition and similarity
    Series: Similarity metric
    """

    sim_table = table[(table.dataset != 'STS') & np.isin(table.conf, ['S-BERT', 'W2V'])]
    sim_table = sim_table.loc[sim_table.groupby(['similarity', 'dataset', 'f_x'])['score'].idxmax()]
    if y_label.lower() == 'pearsonr':
        sim_table = sim_table.groupby(['similarity', 'f_x'])['score'].apply(lambda vs: np.tanh(np.arctanh(vs).mean())).reset_index()
    else:
        sim_table = sim_table.groupby(['similarity', 'f_x'])['score'].mean().reset_index()

    fx_sorter = ['GLOBAL_AVG', 'AVG', 'IND', 'JOINT', 'INF', 'SUM']
    sim_sorter = ['COS', 'DOT', 'ICM', 'EUC']
    sim_sorter = [v for v in sim_sorter if v in table.similarity.unique()]
    fx_sorter = [v for v in fx_sorter if v in table.f_x.unique()]
    sim_table.similarity = sim_table.similarity.astype('category')
    sim_table.similarity.cat.set_categories(sim_sorter, inplace=True)
    sim_table.f_x = sim_table.f_x.astype('category')
    sim_table.f_x.cat.set_categories(fx_sorter, inplace=True)

    sim_table = sim_table.sort_values(['similarity', 'f_x'])

    sim_map = {'ICM': 'ICM (\u03b2=OPT)',
               'EUC': 'EUCLIDEAN (ICM \u03b2=2)',
               'DOT': 'DOT PRODUCT (ICM \u03b2=1)'}
    n_fxs = sim_table.f_x.nunique()
    n_sims = sim_table.similarity.nunique()
    fxs = sim_table.f_x.unique()
    fx_map = {'GLOBAL_AVG': 'S-BERT',
              'AVG': 'W2V\n BEST STRUCTURE\nF_AVG',
              'IND': 'W2V\n BEST STRUCTURE\nF_IND',
              'JOINT': 'W2V\n BEST STRUCTURE\nF_JOINT',
              'INF': 'W2V\n BEST STRUCTURE\nF_INF',
              'SUM': 'W2V\n BEST STRUCTURE\nF_SUM'}
    sims = sim_table.similarity.unique()

    x = np.arange(n_fxs)  # the label locations
    width = 1 / (n_sims + 1)  # the width of the bars

    fig, ax = plt.subplots(figsize=[12, 6])
    cs = ['#FFFFFF'] * 4
    # cs = ['#FE9B3F', '#3E0381', '#A542FC', '#F0A3FC']
    cs = ['#FDCAAA', '#FDAAFC', '#DCAAFD', '#FDAAC4']
    hs = ['///', '\\\\', '++', '*']

    spans = get_spans(width, n_sims)
    rects = get_rects(sim_table, 'similarity', ax, sims, sim_map, cs, hs, width, spans, x)

    ax.set_ylabel(y_label)
    if title_main is None:
        title_main = 'Sentence-Sentence similarity prediction'
    # ax.set_title(f'{title_main} by similarity across composition')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels([fx_map.get(config, config) for config in fxs])
    # ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    ax.legend(loc=legend_loc)

    fig.tight_layout()
    plt.grid()
    if image_name is not None:
        plt.savefig(os.path.join(DATA_PATH, 'images', image_name),
                    bbox_inches='tight')
    else:
        plt.show()


def plot_struct_composition(table, title_main=None, y_label='SpearmanR', image_name=None, legend_loc='upper left'):
    # Horizontal axis: F_INF, COS and specific linguistic structure
    # Vertical axis: Average SpearmanR across dataset for F_INF, COS
    # Series: Linguistic structure

    cos_inf_table = table[(table.similarity == 'COS') & (table.conf == 'INF') & (table.dataset != 'STS')]
    cos_inf_table = cos_inf_table.loc[cos_inf_table.groupby(['dataset', 'struct'])['score'].idxmax()]
    cos_inf_table = cos_inf_table.groupby(['struct'])['score'].mean().reset_index()

    data_sorter = ['MEN', 'SICK', 'STS', 'MSRpar', 'MSRvid', 'answer-answer', 'images', 'track5']
    struct_sorter = ['SEQ', 'SYN', 'DEP']
    struct_sorter = [v for v in struct_sorter if v in table.struct.unique()]
    cos_inf_table.struct = cos_inf_table.struct.astype('category')
    cos_inf_table.struct.cat.set_categories(struct_sorter, inplace=True)
    cos_inf_table = cos_inf_table.sort_values(['struct'])

    n_structs = cos_inf_table.struct.nunique()
    structs = cos_inf_table.struct.unique()

    x = np.arange(1)  # the label locations
    width = 1 / (n_structs + 1)  # the width of the bars

    fig, ax = plt.subplots(figsize=[12, 5])
    cs = ['#FFFFFF'] * 3
    # cs = ['#1095F7', '#FE412E', '#3CCF05']
    cs = ['#FDACAA', '#B4FDAA', '#AAB2FD']
    hs = ['///', '++', '*']

    spans = [-width, 0, width]
    rects = get_rects(cos_inf_table, 'struct', ax, structs, {}, cs, hs, width, spans, x)

    ax.set_ylabel(y_label)
    if title_main is None:
        title_main = 'Sentence-Sentence similarity prediction'
    # ax.set_title(f'{title_main} by structure')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(['INF_COS'])
    ax.legend(bbox_to_anchor=(1.03, 1), loc=legend_loc)

    fig.tight_layout()
    plt.grid()
    if image_name is not None:
        plt.savefig(os.path.join(DATA_PATH, 'images', image_name),
                    bbox_inches='tight')
    else:
        plt.show()


def plot_struct_composition2(table, title_main=None, y_label='SpearmanR', image_name=None):
    # Horizontal axis: F_INF, COS and specific linguistic structure
    # Vertical axis: Average SpearmanR across dataset for F_INF, COS
    # Series: Linguistic structure

    cos_inf_table = table[(table.similarity == 'COS') & (table.dataset != 'STS') & (table.f_x == 'INF')]
    cos_inf_table = cos_inf_table.loc[cos_inf_table.groupby(['dataset', 'struct'])['score'].idxmax()]
    # mean_table = cos_inf_table.groupby(['struct'])['score'].mean()

    data_sorter = ['MSRpar', 'MSRvid', 'answer-answer', 'images', 'track5', 'SICK', 'MEN']
    struct_sorter = ['SEQ', 'SYN', 'DEP']
    data_sorter = [v for v in data_sorter if v in table.dataset.unique()]
    struct_sorter = [v for v in struct_sorter if v in table.struct.unique()]
    cos_inf_table.struct = cos_inf_table.struct.astype('category')
    cos_inf_table.struct.cat.set_categories(struct_sorter, inplace=True)
    cos_inf_table.dataset = cos_inf_table.dataset.astype('category')
    cos_inf_table.dataset.cat.set_categories(data_sorter, inplace=True)
    cos_inf_table = cos_inf_table.sort_values(['dataset', 'struct'])

    data_map = {'MEN': 'DEF2DEF',
                'MSRpar': 'STS-MSRpar',
                'MSRvid': 'STS-MSRvid',
                'answer-answer': 'STS-answer-answer',
                'images': 'STS-images',
                'track5': 'STS-track5'}
    struct_map = {'SYN': 'CONST'}

    n_structs = cos_inf_table.struct.nunique()
    structs = cos_inf_table.struct.unique()
    n_datasets = cos_inf_table.dataset.nunique()
    datasets = cos_inf_table.dataset.unique().tolist()

    x = np.arange(n_datasets)    # the label locations
    width = 1 / (n_structs + 1)  # the width of the bars

    fig, ax = plt.subplots(figsize=[12, 5])
    rects = []
    # cs = ['#0294E3', '#FB4E56', '#23D207']
    cs = ['#FFFFFF'] * 3
    cs = ['#FDACAA', '#B4FDAA', '#AAB2FD']
    hs = ['///', '++', '*']

    spans = get_spans(width, n_structs)
    rects = get_rects(cos_inf_table, 'struct', ax, structs, struct_map, cs, hs, width, spans, x)

    ax.set_ylabel(y_label)
    if title_main is None:
        title_main = 'Sentence-Sentence similarity prediction'
    # ax.set_title(f'{title_main} by structure across datasets')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels([data_map.get(d, d) for d in datasets])
    # ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
    ax.legend(loc='upper right')

    fig.tight_layout()
    plt.grid()
    if image_name is not None:
        plt.savefig(os.path.join(DATA_PATH, 'images', image_name),
                    bbox_inches='tight')
    else:
        plt.show()


def tokenize(tokenizer, sent, max_len):
    return tokenizer.encode_plus(
                                    sent,
                                    add_special_tokens=True,
                                    max_length=max_len,
                                    return_token_type_ids=False,
                                    padding=False,
                                    return_attention_mask=True,
                                    return_tensors='pt',
                                    truncation=True
                                 )


def get_comparison_words(sentences, word2vecs, stopwords=[]):
    # Ensure all words in sentences have a word2vec instance
    available_words = [w for w in word2vecs.vocab.keys()
                       if not re.search('[^a-z]', w)
                       and w not in sentences.index.tolist()
                       and wn.synsets(w)]

    print('Number of lowercase words that do not appear in our corpus and have a synset in wordnet' +
          f': {len(available_words)}')

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(w, pos=wn.NOUN),
                                                                             pos=wn.VERB), pos=wn.ADV), pos=wn.ADJ)
              for w in available_words]
    lemmas = set(lemmas)
    print(f'Number of lemmas from the available words: {len(lemmas)}')

    filt_lemmas = [w for w in lemmas if w not in stopwords and wn.synsets(w) and w in word2vecs.vocab.keys()]
    print(f'Number of lemmas that are not stopwords and still appear in word2vec: {len(filt_lemmas)}')

    # Extract 30.000 random words from the set of available lemmas
    np.random.seed(RANDOM_SEED)

    n_words = 30000
    return np.array(filt_lemmas)[np.random.permutation(len(filt_lemmas))[:n_words]].tolist()


def update_dict(prev_dict, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, dict):
            if k not in prev_dict or not isinstance(prev_dict[k], dict):
                prev_dict[k] = v
            else:
                update_dict(prev_dict[k], v)
        else:
            prev_dict[k] = v

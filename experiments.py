import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import functions, similarity
from models import BertForMaskedLMSoftmax
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from utils.composition import sents2sims, sent2sims_SBERT, sent2sims_LM
from utils.config_params import *
from utils.inout import read_comparison_words, read_stopwords, DATA_PATH
from utils.stats import dependent_corr


def init_transformers(device, use_bert=USE_BERT, use_sbert=USE_SBERT, use_gpt2=USE_GPT2, bert_modes=BERT_MODES,
                      gpt_modes=GPT_MODES):
    models = {}
    tokenizers = {}
    modes = {}
    if use_bert:
        # Initialize BERT Tokenizer and pre-trained model
        bert_model_name = 'bert-base-uncased'

        tokenizers['BERT'] = AutoTokenizer.from_pretrained(bert_model_name)
        models['BERT'] = AutoModel.from_pretrained(bert_model_name)
        # models['BERT'] = BertForMaskedLMSoftmax(bert_model_name)
        models['BERT'] = models['BERT'].to(device)
        models['BERT'] = models['BERT'].eval()
        modes['BERT'] = bert_modes

    if use_sbert:
        # Initialize Sentence-BERT model
        models['S-BERT'] = SentenceTransformer('bert-base-nli-mean-tokens')

    if use_gpt2:
        # Initialize GPT-2 model
        gpt_model_name = 'gpt2'

        tokenizers['GPT'] = AutoTokenizer.from_pretrained(gpt_model_name)
        models['GPT'] = AutoModel.from_pretrained(gpt_model_name)
        models['GPT'] = models['GPT'].to(device)
        models['GPT'] = models['GPT'].eval()
        modes['GPT'] = gpt_modes

    return tokenizers, models, modes


def fset_experiments(sents, exps, word_vecs, gamma, K=1, f_comp=functions._comp, f_sim=cosine_similarity, stopwords=[],
                     f_corr=spearmanr):
    """
        First set of experiments:
        -   Datasets: MEN, SICK and STS
        -   Models: BERT, S-BERT, F_SUM, F_AVG, F_IND, F_JOINT, F_INF
        -   Linguistic Structures: Sequential, Constituents and Dependencies
    """
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    starts = ['left', 'right']
    criterias = ['seq', 'syn', 'dep']
    configs = {
        # 'SUM': (1, -2),
        'AVG': (1 / 4, -1 / 2),
        'IND': (1, 0),
        'JOINT': (1, 1),
        'INF': (1, None)
        }
    results = {}

    tokenizers, models, modes = init_transformers(device)

    for dataset in exps:
        results[dataset] = {}
        print(f'Making experiments for {dataset} dataset with {exps[dataset].shape[0]} experiments...')

        for start in starts:
            print('+++ Start: ' + start)
            results[dataset][start] = {}

            if USE_W2V:
                for cri in criterias:
                    print('+++++++ Criteria: ' + cri)
                    if cri == 'dep':
                        col = cri + '_' + start
                    else:
                        col = cri

                    sents[dataset][col] = sents[dataset][col].apply(functions.to_lowercase)
                    results[dataset][start][cri] = {}
                    for conf in configs:
                        alpha = configs[conf][0]
                        beta = configs[conf][1]

                        sims = sents2sims(sents[dataset][[col]], exps[dataset], word_vecs, start, alpha, beta, gamma, K,
                                          f_comp, f_sim, stopwords)

                        results[dataset][start][cri][conf] = f_corr(sims.values, exps[dataset]['gs'].values)[0]

            if not stopwords:
                # BERT Experiments
                # The maximum number of tokens admitted by BERT is 512
                if USE_BERT:
                    print('+++ BERT')
                    results[dataset][start]['BERT'] = get_lm_results(sents, exps, dataset,
                                                                     tokenizers['BERT'], models['BERT'], modes['BERT'],
                                                                     f_sim, device, start, f_corr=f_corr)

                if USE_GPT2:
                    print('+++ GPT 2')
                    results[dataset][start]['GPT'] = get_lm_results(sents, exps, dataset,
                                                                    tokenizers['GPT'], models['GPT'], modes['GPT'],
                                                                    f_sim, device, start, lm_layer=2, f_corr=f_corr)

        if USE_W2V:
            print('+++ SUM')
            sum_sims = sents2sims(sents[dataset][['seq']], exps[dataset], word_vecs, 'left', None, None, None, None,
                                  functions._sum, f_sim, stopwords)

            results[dataset]['SUM'] = f_corr(sum_sims.values, exps[dataset]['gs'].values)[0]

        if not stopwords:
            # SBERT Experiments
            if USE_SBERT:
                print('+++ S-BERT')
                s_bert_sims = sent2sims_SBERT(sents[dataset][['text']], exps[dataset], models['S-BERT'], f_sim)
                results[dataset]['S-BERT'] = f_corr(s_bert_sims, exps[dataset]['gs'].values)[0]

    return results


def get_lm_sims(sents, exps, dataset, tokenizer, model, modes, f_sim, device, start, lm_layer=None, verbose=True):
    token_lens = sents[dataset].text.apply(lambda x: len(tokenizer.encode(x,
                                                                          max_length=512,
                                                                          truncation=True))).to_numpy()
    max_len = np.max(token_lens)
    with torch.no_grad():
        sims = sent2sims_LM(sents[dataset][['text']], exps[dataset],
                            tokenizer, model, max_len, modes, f_sim, device, start, lm_layer, verbose)
    return sims

def get_lm_results(sents, exps, dataset, tokenizer, model, modes, f_sim, device, start, lm_layer=None, f_corr=spearmanr):
    sims = get_lm_sims(sents, exps, dataset, tokenizer, model, modes, f_sim, device, start, lm_layer)

    results = {}
    for mode in modes:
        results[mode] = f_corr(sims[mode].values, exps[dataset]['gs'].values)[0]

    return results


def sset_experiments(sents, word_vecs, gamma, K=1, f_comp=functions._comp, f_sim=cosine_similarity, stopwords=[]):
    """
        Second set of experiments:
        -   Dataset: WORD-DEF (Words obtained from MEN's dataset)
        -   Models: BERT, S-BERT, F_SUM, F_AVG, F_IND, F_JOINT, F_INF
        -   Linguistic Structures: Sequential, Constituents and Dependencies
    """
    exp = 'MEN'
    # Word2Vec contains the american variants of the following words
    change_words = {'colour': 'color', 'grey': 'gray', 'harbour': 'harbor', 'theatre': 'theater'}
    results = {exp: {}}
    orig_words = [w if w not in change_words else change_words[w] for w in sents.index.tolist()]
    comp_words = read_comparison_words()
    if USE_W2V:
        comp_vws = np.array([word_vecs[w] for w in comp_words])
        orig_vws = np.array([word_vecs[w] for w in orig_words])

    starts = ['left', 'right']
    criterias = ['seq', 'syn', 'dep']
    configs = {  # 'SUM': (1, -2),
        'AVG': (1 / 4, -1 / 2),
        'IND': (1, 0),
        'JOINT': (1, 1),
        'INF': (1, None)
        }

    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizers, models, modes = init_transformers(device)

    for start in starts:
        print('+++ Start: ' + start)
        results[exp][start] = {}

        if USE_W2V:
            for cri in criterias:
                print('+++++++ Criteria: ' + cri)
                if cri == 'dep':
                    col = cri + '_' + start
                else:
                    col = cri

                sents[col] = sents[col].apply(functions.to_lowercase)

                results[exp][start][cri] = {}
                for conf in configs:
                    alpha = configs[conf][0]
                    beta = configs[conf][1]

                    roc_auc = similarity.compare_wds(sents, col, comp_vws, orig_vws, word_vecs,
                                                     start, alpha, beta, gamma, K=K,
                                                     comp=f_comp, f_sim=f_sim, stopwords=stopwords)

                    results[exp][start][cri][conf] = roc_auc

        if not stopwords:
            max_len = 512
            if USE_BERT:
                with torch.no_grad():
                    print('+++ BERT')
                    results[exp][start]['BERT'] = {}
                    for mode in modes['BERT']:
                        print('+++++++ Mode: ' + mode)
                        bert_result = similarity.compare_wds_LM(sents, 'text', comp_words, orig_words,
                                                                tokenizers['BERT'], models['BERT'],
                                                                max_len, mode, f_sim, start)

                        results[exp][start]['BERT'][mode] = bert_result

            if USE_GPT2:
                with torch.no_grad():
                    print('+++ GPT 2')
                    results[exp][start]['GPT'] = {}
                    for mode in modes['GPT']:
                        print('+++++++ Mode: ' + mode)
                        gpt_result = similarity.compare_wds_LM(sents, 'text', comp_words, orig_words,
                                                               tokenizers['GPT'], models['GPT'],
                                                               max_len, mode, f_sim, start, lm_layer=2)

                        results[exp][start]['GPT'][mode] = gpt_result

    if USE_W2V:
        print('+++ SUM')
        sum_result = similarity.compare_wds(sents, 'seq', comp_vws, orig_vws, word_vecs,
                                            'left', 1, 0, 1, K=K,
                                            comp=functions._sum, f_sim=f_sim, stopwords=stopwords)

        results[exp]['SUM'] = sum_result

        del comp_vws, orig_vws, word_vecs

    if not stopwords:
        if USE_SBERT:
            print('+++ S-BERT')
            sbert_result = similarity.compare_wds_sbert(sents, 'text', comp_words, orig_words,
                                                        models['S-BERT'], f_sim)

            results[exp]['S-BERT'] = sbert_result

    return results


def perform_stat_analysis(sents, exps, word_vecs, conf_df, **kwargs):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f_sims = {'cos': cosine_similarity,
              'euc': functions.euclidean_score,
              'dot': functions.dot_product,
              'icm': functions.icm}
    f_corr = kwargs['f_corr']
    f_comp = kwargs['f_comp']

    results = {}

    tokenizers, models = None, None
    # tokenizers, models, _ = init_transformers(device)

    # 1st analysis: Language Models
    table1 = conf_df[(conf_df.similarity == 'COS') & (conf_df.struct == 'SEQ') & (conf_df.dataset != 'STS')]
    table1 = table1.loc[table1.groupby(['dataset', 'conf'])['score'].idxmax()]
    table1 = table1.sort_values(['dataset', 'conf'])

    lmc, lms = extract_corr_stats(sents, exps, word_vecs, table1, f_sims['cos'], f_comp, f_corr, tokenizers, models, device,
                                  table_key='conf')

    # 2nd analysis: Composition functions
    table2 = conf_df[(conf_df.similarity == 'COS') & (conf_df.conf == 'W2V') & (conf_df.dataset != 'STS')]
    table2 = table2.loc[table2.groupby(['dataset', 'f_x'])['score'].idxmax()]
    table2 = table2.sort_values(['dataset', 'f_x'])

    fnc, fns = extract_corr_stats(sents, exps, word_vecs, table2, f_sims['cos'], f_comp, f_corr, tokenizers, models, device,
                                  table_key='f_x')

    # 3rd analysis: Similarity functions
    table3 = conf_df[(conf_df.dataset != 'STS') & np.isin(conf_df.conf, ['S-BERT', 'W2V'])]
    table3 = table3.loc[table3.groupby(['similarity', 'dataset', 'f_x'])['score'].idxmax()]

    sic, sis = extract_corr_stats(sents, exps, word_vecs, table3, f_sims, f_comp, f_corr, tokenizers, models, device,
                                      table_key=['similarity', 'f_x'])

    sims = [k.upper() for k in f_sims.keys()]
    scores = pd.concat([c.loc[[f'{s}-INF' for s in sims], ['GS']] for c in sic], axis=1)
    rel = sis[0].iloc[1::2].copy()
    rel[:] = 0
    rel = rel.astype(int)
    for ddata in sis:
        cond1 = (ddata.iloc[1::2] <= 0.05).values
        cond2 = (ddata.iloc[::2] > 0).values
        # rel += (cond1 & cond2).astype(int)
        rel += cond2.astype(int)

    # 4th analysis: Linguistic structures
    table4 = conf_df[(conf_df.similarity == 'COS') & (conf_df.dataset != 'STS') & (conf_df.f_x == 'INF')]
    table4 = table4.loc[table4.groupby(['dataset', 'struct'])['score'].idxmax()]
    table4 = table4.sort_values(['dataset', 'struct'])

    stc, sts = extract_corr_stats(sents, exps, word_vecs, table4, f_sims['cos'], f_comp, f_corr, tokenizers, models, device,
                                  table_key='struct')

    return results


def list2str(l):
    if isinstance(l, list):
        return '-'.join(l)
    else:
        return l


def extract_corr_stats(sents, exps, word_vecs, table, f_sim, f_comp, f_corr, tokenizers, models, device, table_key):
    corrs_out = []
    stats_out = []
    for dataset, d_configs in table.groupby('dataset'):
        corrs_file = os.path.join(DATA_PATH, 'results', f'{dataset}_corrs_{list2str(table_key)}.csv')
        stats_file = os.path.join(DATA_PATH, 'results', f'{dataset}_stats_{list2str(table_key)}.csv')
        keys = [list2str(k) for k in d_configs[table_key].values.tolist()]
        if not os.path.isfile(corrs_file):
            print(f'Computing correlations for {dataset} dataset and {table_key} interest')
            sims = get_similarities(d_configs, sents, exps, word_vecs, dataset, f_sim, f_comp, tokenizers, models,
                                            device, table_key)
            sims = pd.DataFrame(sims)
            sims['GS'] = exps[dataset]['gs'].values
            corrs = sims.corr(method=f_corr.__name__[:-1])
            corrs.to_csv(corrs_file, index=True)
        else:
            print(f'Reading correlations for {dataset} dataset and {table_key} interest')
            corrs = pd.read_csv(corrs_file, index_col=0)

        corrs_out.append(corrs)
        # print(corrs)

        if not os.path.isfile(stats_file):
            print(f'Computing statistical significance for {dataset} dataset and {table_key} interest')
            stats = {}
            for i in range(len(keys)):
                gs_corr_x = corrs.loc[keys[i], 'GS']
                for j in range(len(keys)):
                    gs_corr_y = corrs.loc[keys[j], 'GS']
                    xy_corr = corrs.loc[keys[i], keys[j]]
                    t2, p = dependent_corr(gs_corr_x, gs_corr_y, xy_corr, exps[dataset].shape[0],
                                           twotailed=True, method='steiger')
                    stats[(keys[i], 't2')] = stats.get((keys[i], 't2'), []) + [t2]
                    stats[(keys[i], 'p')] = stats.get((keys[i], 'p'), []) + [p]
            stats = pd.DataFrame.from_dict(stats, orient='index', columns=keys)
            stats.to_csv(stats_file, index=True)
        else:
            print(f'Reading statistical significance for {dataset} dataset and {table_key} interest')
            stats = pd.read_csv(stats_file, index_col=0)

        stats_out.append(stats)
        print(stats.round(4))

    return corrs_out, stats_out


def get_similarities(config_df, sents, exps, word_vecs, dataset, f_sims, f_comp, tokenizers, models, device, table_key):
    start_map = {'L2R': 'left', 'R2L': 'right'}
    aggr_params = {
        # 'SUM': (1, -2),
        'AVG': (1 / 4, -1 / 2),
        'IND': (1, 0),
        'JOINT': (1, 1),
        'INF': (1, None)
    }

    similarities = {}
    for i, row in config_df.iterrows():
        if isinstance(f_sims, dict):
            f_sim = f_sims[row.similarity.lower()]
        else:
            f_sim = f_sims
        stopwords = []
        start = start_map.get(row.direction, None)
        if row.stopwords == 'noSW':
            stopwords = read_stopwords()
        if row.conf == 'W2V':
            if row.f_x in aggr_params:
                cri = row.struct.lower().replace('const', 'syn')
                if row.struct.lower() == 'dep':
                    col = row.struct.lower() + '_' + start
                else:
                    col = cri
                alpha = aggr_params[row.f_x][0]
                beta = aggr_params[row.f_x][1]
                sims = sents2sims(sents[dataset][[col]], exps[dataset], word_vecs, start, alpha, beta, GAMMA, K,
                                  f_comp, f_sim, stopwords)
            else:
                # The SUM case
                sims = sents2sims(sents[dataset][['seq']], exps[dataset], word_vecs, 'left', None, None, None, None,
                                  functions._sum, f_sim, stopwords)
        elif row.conf in ['BERT', 'BERT_CLS']:
            sims = get_lm_sims(sents, exps, dataset, tokenizers['BERT'], models['BERT'], [row.f_x], f_sim, device,
                               start, verbose=False)[row.f_x]
        elif row.conf == 'GPT':
            sims = get_lm_sims(sents, exps, dataset, tokenizers['GPT'], models['GPT'], [row.f_x], f_sim, device,
                               start, lm_layer=2, verbose=False)[row.f_x]
        elif row.conf == 'S-BERT':
            sims = sent2sims_SBERT(sents[dataset][['text']], exps[dataset], models['S-BERT'], f_sim)
        else:
            raise NotImplementedError('Please, provide a valid model!')

        if isinstance(table_key, list):
            k = list2str(row[table_key].values.tolist())
        else:
            k = row[table_key]
        similarities[k] = sims.values

    return similarities

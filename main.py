import json
import os

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

from experiments import fset_experiments, sset_experiments, perform_stat_analysis
from utils import functions
from utils.functions import *
from utils.inout import *
from utils.config_params import *


def main_perform_experiments():
    # Set parameters
    f_comp = functions._comp
    f_sims = {'cos': cosine_similarity,
              'euc': functions.euclidean_score,
              'dot': functions.dot_product,
              'icm': functions.icm}
    f_corr = pearsonr  # spearmanr

    sentences, experiments, _ = read_data()

    if STS_ONLY:
        experiments['STS']['source'] = experiments['STS']['text1'].apply(lambda s: re.sub('\..*', '', s))
        # Decompose the STS dataset into multiple subsets
        experiments = {idx: subset.reset_index(drop=True) for idx, subset in experiments['STS'].groupby('source')}
        sentences = {key: sentences['STS'].loc[set(subset[['text1', 'text2']].values.flatten())]
                     for key, subset in experiments.items()}

    for use_stopwords in [True, False]:
        if STS_ONLY:
            filename = 'finalSTS_'
        else:
            filename = 'final_'

        if use_stopwords:
            stopwords = []
            filename += 'withSW_'
        else:
            stopwords = read_stopwords()
            filename += 'withoutSW_'

        filename += 'xxx'

        for sim_str, f_sim in f_sims.items():
            filename = filename[:-3] + sim_str

            print(f'\nWorking on experiment {filename}...')

            if USE_W2V:
                word2vec = read_word2vec()
            else:
                word2vec = None

            print(f'\nInitiating first set of experiments...')
            results = fset_experiments(sentences, experiments, word2vec, GAMMA, K, f_comp, f_sim, stopwords, f_corr)
            print(f'\nWriting results corresponding to the first set of experiments...')
            write_results(results, f"{filename}_{f_corr.__name__}_1.json")

            if not STS_ONLY:
                print(f'\nInitiating second set of experiments...')
                results = sset_experiments(sentences['MEN'], word2vec, GAMMA, K, f_comp, f_sim, stopwords)
                print(f'\nWriting results corresponding to the first set of experiments...')
                write_results(results, filename + '_2.json')


def main_statistical_analysis():
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    metrics = ['euc', 'dot', 'cos', 'icm']
    sw_modes = {'withSW': 'SW', 'withoutSW': 'noSW'}
    f_corr = pearsonr

    table, _ = get_results(metrics, sw_modes, suffix=f'{f_corr.__name__}_')
    sentences, experiments, _ = read_data()
    experiments['STS']['source'] = experiments['STS']['text1'].apply(lambda s: re.sub('\..*', '', s))
    sts_experiments = {idx: subset.reset_index(drop=True) for idx, subset in experiments['STS'].groupby('source')}
    experiments.update(sts_experiments)
    sentences.update({key: sentences['STS'].loc[set(subset[['text1', 'text2']].values.flatten())]
                      for key, subset in sts_experiments.items()})
    experiments.pop('STS')
    sentences.pop('STS')

    # Preprocessing to the sentences
    for dataset in table.dataset.unique():
        if dataset != 'STS':
            for start in ['left', 'right']:
                for cri in ['seq', 'syn', 'dep']:
                    if cri == 'dep':
                        col = cri + '_' + start
                    else:
                        col = cri

                    sentences[dataset][col] = sentences[dataset][col].apply(functions.to_lowercase)

    f_comp = functions._comp

    word_vecs = []
    # if 'W2V' in table.conf.unique():
    #     word_vecs = read_word2vec()

    perform_stat_analysis(sentences, experiments, word_vecs, table, f_comp=f_comp, f_corr=f_corr)

    print('Statistical analysis completed...')


def main_view_results():
    metrics = ['euc', 'dot', 'cos', 'icm']
    sw_modes = {'withSW': 'SW', 'withoutSW': 'noSW'}
    corr = 'pearson'  # spearman

    # results_file = 'final_withSW_cos_1.json'
    fset_table, sset_table = get_results(metrics, sw_modes, suffix=f'{corr}r_')

    plot_steps = [1, 2, 3, 4]
    save_imgs = False

    if 1 in plot_steps:
        # Figures 1 and 2
        image_name = f'{corr}_comp_dataset_lm.png' if save_imgs else None
        plot_comp_dataset_lm(fset_table, title_main=None, y_label=corr.upper(), image_name=image_name,
                             dataset_label=None, legend_loc='upper right')
        image_name = 'auc-roc_comp_dataset_lm.png' if save_imgs else None
        plot_comp_dataset_lm(sset_table, title_main=None, y_label='AUC-ROC', image_name=image_name,
                             dataset_label='WORD2DEF', legend_loc='lower right')

    if 2 in plot_steps:
        # Figures 3 and 4
        image_name = f'{corr}_comp_dataset_w2v.png' if save_imgs else None
        plot_comp_dataset_w2v(fset_table, title_main=None, y_label=corr.upper(), image_name=image_name,
                              dataset_label=None, legend_loc='lower right')
        image_name = 'auc-roc_comp_dataset_w2v.png' if save_imgs else None
        plot_comp_dataset_w2v(sset_table, title_main=None, y_label='AUC-ROC', image_name=image_name,
                              dataset_label='WORD2DEF', legend_loc='lower right')

    if 3 in plot_steps:
        # Figures 5 and 6
        y_label = f'AVERAGE {corr.upper()} ACROSS STS, DEF2DEF AND SICK DATASETS'
        image_name = f'{corr}_similarity_comp.png' if save_imgs else None
        plot_similarity_composition(fset_table, title_main=None, y_label=y_label, image_name=image_name,
                                    legend_loc='upper right')
        y_label = 'AUC-ROC AT WORD2DEF DATASET'
        image_name = 'auc-roc_similarity_comp.png' if save_imgs else None
        plot_similarity_composition(sset_table, title_main=None, y_label=y_label, image_name=image_name,
                                    legend_loc='lower right')

    if 4 in plot_steps:
        # Figure 7
        y_label = f'{corr.upper()} CORRELATION WITH W2V + F_INF + COS'
        image_name = f'{corr}_struct_comp.png' if save_imgs else None
        plot_struct_composition2(fset_table, title_main=None, y_label=y_label, image_name=image_name)
        # plot_struct_composition2(sset_table, title_main=None, y_label='AUC-ROC', image_name=None)


def main_latex():
    metrics = ['euc', 'dot', 'cos', 'icm']
    sw_modes = {'withSW': 'SW', 'withoutSW': 'noSW'}

    fset_table, sset_table = get_results(metrics, sw_modes)
    transform_table(fset_table)
    transform_table(sset_table)


def transform_table(table):
    def extract_comps(g):
        return pd.Series({r.F_x: r.Score for i, r in g.iterrows()})

    table.fillna('-', inplace=True)

    table['conf'].replace('BERT_CLS', 'BERT', inplace=True)

    aux = table.loc[np.bitwise_and(table.f_x == 'SUM', table.direction == '-')]
    table.loc[np.bitwise_and(table.f_x == 'SUM', table.direction == '-'), 'direction'] = 'L2R'
    aux.direction = 'R2L'
    table = table.append(aux, ignore_index=True)

    table['Other'] = table[['direction', 'stopwords']].agg(' - '.join, axis=1)
    table.drop(['stopwords', 'direction'], axis=1, inplace=True)
    table.columns = ['Similarity', 'Dataset', 'Structure', 'Embedding', 'score', 'f_x', 'Other']
    indices = ['Similarity', 'Dataset', 'Structure', 'Embedding', 'Other']
    # table.set_index(indices, drop=True, inplace=True)
    table = table.pivot_table(values='score', index=indices, columns='f_x', aggfunc='first')
    print(table.to_latex(na_rep='-', float_format='%.4f', multirow=True, longtable=True))


def test():
    sw_mode = 'withSW'
    metric = 'cos'
    filename = f'{sw_mode}_{metric}_'
    new_results = read_results(f'finalSTS_{filename}1.json')
    old_results = read_results(f'final/finalSTS_{filename}1.json')
    update_dict(old_results, new_results)

    with open(os.path.join(DATA_PATH, 'results', f'finalSTS_{filename}1.json'), 'w') as fp:
        json.dump(old_results, fp)

    print('hey')


if __name__ == '__main__':
    # save_comparison_words()
    # main_perform_experiments()
    main_view_results()
    # main_latex()
    # main_statistical_analysis()

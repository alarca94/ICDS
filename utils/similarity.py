import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils import functions
from utils import composition
from utils.config_params import RANDOM_SEED


def get_numerator(idx1, idx2, iqs):
    return iqs.loc[idx1] + iqs.loc[idx2]


def get_denominator(idx1, idx2, numerator, comps):
    return numerator - np.dot(comps.loc[idx1].reshape(1, -1), comps.loc[idx2].reshape(-1, 1))


def get_partial_fraction(idx1, idx2, iqs, comps):
    num = get_numerator(idx1, idx2, iqs)
    return (num, get_denominator(idx1, idx2, num, comps))


def estimate_beta_icm(comps, exp_df):
    iqs = comps.apply(lambda comp: np.sum(comp ** 2))
    partials = exp_df.apply(lambda row: get_partial_fraction(row.text1, row.text2, iqs, comps), axis=1).tolist()
    num, den = list(zip(*partials))
    return np.sum(num) / np.sum(den)


def compute_sims(comps, exp_df, f_sim):
    if f_sim == functions.icm:
        icm_beta = estimate_beta_icm(comps, exp_df)
        assert 1 <= icm_beta <= 2
        sim_params = {'a1': 1, 'a2': 1, 'b': icm_beta}
    else:
        sim_params = None

    return exp_df.apply(lambda row: composition.compare_texts(row.text1, row.text2, comps, f_sim, sim_params), axis=1)


def estimate_betas_icm_mrr(orig_vws, comp_vws, comps):
    # Compute the information quantities of each document
    iqs_orig_vws = np.sum(orig_vws ** 2, axis=1, keepdims=True)
    iqs_comp_vws = np.sum(comp_vws ** 2, axis=1, keepdims=True)
    iqs_comps = np.sum(comps ** 2, axis=1, keepdims=True)

    # Compute the numerator (sum of information quantities of each pair of documents)
    iq_pairs = iqs_comp_vws + iqs_comps.T
    iq_pairs = np.vstack((iq_pairs, (iqs_orig_vws + iqs_comps).T))  # Shape: [COMP_VWS + ORIG_VWS, COMPS]

    # Compute the joint information quantity of all document pairs
    dot_product = np.dot(comp_vws, comps.T)
    dot_product = np.vstack((dot_product, np.sum(np.multiply(orig_vws, comps), axis=1, keepdims=True).T))

    # Return the beta estimation for each composition (compared to all words)
    return (np.sum(iq_pairs, axis=0) / np.sum(iq_pairs - dot_product, axis=0)).reshape(-1, 1)


def compute_sims_ovr(orig_vws, comp_vws, comps, f_sim):
    """
        Computation of the similarities of each composition in comps versus all comparison words [comp_vws] and its
        respective defined word in orig_vws using f_sim as the similarity metric
    """
    if f_sim == functions.icm:
        icm_betas = estimate_betas_icm_mrr(orig_vws, comp_vws, comps)
        # assert ((1 <= icm_betas).all() and (icm_betas <= 2).all())
        sims = f_sim(comps, comp_vws, c=icm_betas)
        extra_sims = np.diag(f_sim(comps, orig_vws, c=icm_betas)).reshape(-1, 1)
    else:
        sims = f_sim(comps, comp_vws)
        extra_sims = np.diag(f_sim(comps, orig_vws)).reshape(-1, 1)

    return np.hstack((sims, extra_sims))


def compute_sims_ns(orig_vws, comp_vws, comps, f_sim):
    """
        Compute the similarity between the definition's composition and both positive and negative word samplings
        Positive Sampling: Defined word
        Negative Sampling: 100 random words from all comparison words
    """
    np.random.seed(RANDOM_SEED)

    n_neg_samples = 100
    pred_sims = []
    gs_sims = []
    for def_id in range(comps.shape[0]):
        random_ws = np.random.permutation(comp_vws.shape[0])[:n_neg_samples]
        comp_vws_aux = comp_vws[random_ws, :]
        pred_sims.append(compute_sims_ovr(orig_vws[[def_id], :], comp_vws_aux, comps[[def_id], :], f_sim))
        gs_sims.extend([0] * n_neg_samples + [1])

    return np.hstack(pred_sims)[0].tolist(), gs_sims


def compare_wds(df, col, comp_vws, orig_vws, word_vecs, start, alpha, beta, gamma, K=1,
                comp=functions._comp, f_sim=cosine_similarity, stopwords=[]):
    comps = df.apply(lambda x: functions.compose(x[col], word_vecs,
                                                 alpha=alpha, beta=beta, gamma=gamma, K=K, start=start,
                                                 comp_func=comp, stopwords=stopwords),
                     axis=1).tolist()
    comps = np.array([c if c is not None else np.zeros((300,)) for c in comps])
    pred_sims, gs_sims = compute_sims_ns(orig_vws, comp_vws, comps, f_sim)

    return roc_auc_score(gs_sims, pred_sims)


def compare_wds_rank(df, col, comp_vws, orig_vws, word_vecs, start, alpha, beta, gamma, K=1,
                     comp=functions._comp, f_sim=cosine_similarity, stopwords=[]):
    comps = df.apply(lambda x: functions.compose(x[col], word_vecs,
                                                 alpha=alpha, beta=beta, gamma=gamma, K=K, start=start,
                                                 comp_func=comp, stopwords=stopwords),
                     axis=1).tolist()

    comps = np.array([c if c is not None else np.zeros((300,)) for c in comps])
    sims_rank = np.argsort(compute_sims_ovr(orig_vws, comp_vws, comps, f_sim), axis=1)

    return 1 / (sims_rank.shape[1] - np.argwhere(sims_rank == sims_rank.shape[1] - 1)[:, 1])


def compare_wds_LM(df, col, comp_words, orig_words, tokenizer, model, max_len, mode, f_sim, start, lm_layer=None):
    comp_vws = [composition.get_LM_composition(w, tokenizer, model, max_len, mode, start=start, lm_layer=lm_layer)
                for w in tqdm(comp_words)]
    orig_vws = [composition.get_LM_composition(w, tokenizer, model, max_len, mode, start=start, lm_layer=lm_layer)
                for w in orig_words]
    comps = [composition.get_LM_composition(sent, tokenizer, model, max_len, mode, start=start, lm_layer=lm_layer)
             for sent in df[col].values]
    comp_vws, orig_vws, comps = np.array(comp_vws), np.array(orig_vws), np.array(comps)
    pred_sims, gs_sims = compute_sims_ns(orig_vws, comp_vws, comps, f_sim)
    return roc_auc_score(gs_sims, pred_sims)


def compare_wds_sbert(df, col, comp_words, orig_words, model, f_sim):
    comp_vws = model.encode(comp_words)
    orig_vws = model.encode(orig_words)
    comps = model.encode(df[col].tolist())
    pred_sims, gs_sims = compute_sims_ns(orig_vws, comp_vws, comps, f_sim)
    return roc_auc_score(gs_sims, pred_sims)


def estimate_beta_icm_wrong_1(comps):
    # Compute the information quantities of each document
    iqs = np.sum(comps ** 2, axis=1, keepdims=True)

    # Compute the numerator (sum of information quantities of each pair of documents)
    iq_pairs = np.triu(iqs + iqs.T, 1)
    sum_iq_pairs = np.sum(iq_pairs)

    # Compute the joint information quantity of all document pairs
    jiqs = iq_pairs - np.triu(np.dot(comps, comps.T), 1)
    sum_jiqs = np.sum(jiqs)

    # Return the beta estimation
    return sum_iq_pairs / sum_jiqs


def estimate_beta_icm_wrong_2(comps):
    # Compute the information quantities of each document
    iqs = np.sum(comps ** 2, axis=1, keepdims=True)
    triu_ixs = np.triu_indices(comps.shape[0], k=1)

    # Compute the numerator (sum of information quantities of each pair of documents)
    iq_pairs = (iqs + iqs.T)[triu_ixs]

    # Compute the joint information quantity of all document pairs
    jiqs = iq_pairs - np.dot(comps, comps.T)[triu_ixs]

    # Return the beta estimation
    return np.mean(iq_pairs / jiqs)

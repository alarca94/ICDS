import numpy as np
import pandas as pd
import torch
from transformers import BertModel

from utils import functions, similarity
from sklearn.metrics.pairwise import cosine_similarity


def compute_comps(sents_df, word_vecs, start, alpha, beta, gamma, K=1, comp=functions._comp, stopwords=[]):
    comps = sents_df.iloc[:, 0].apply(lambda s: functions.compose(s, word_vecs, alpha=alpha, beta=beta, gamma=gamma,
                                                                  K=K, start=start, comp_func=comp,
                                                                  stopwords=stopwords))

    # Fill None values with a vector of zeros (in case the composition could not find any representation)
    if any(comps.isna()):
        comps = comps.apply(lambda r: r if r is not None else np.zeros((word_vecs.vector_size,)))

    return comps


def compare_texts(t1, t2, comps, comp_metric=cosine_similarity, comp_params=None):
    comp1 = comps.loc[t1].reshape(1, -1)
    comp2 = comps.loc[t2].reshape(1, -1)
    if np.isnan(comp1).any():
        print(f'{t1} comp contains at least one nan')
        print(comp1)

    if np.isnan(comp2).any():
        print(f'{t2} comp contains at least one nan')
        print(comp2)

    if comp_metric == functions.icm:
        return comp_metric(comp1, comp2, comp_params['b'], comp_params['a1'], comp_params['a2'])[0, 0]
    else:
        return comp_metric(comp1, comp2)[0, 0]


def sents2sims(sents_df, exp_df, word_vecs, start, alpha, beta, gamma, K, f_comp, f_sim, stopwords):
    comps = compute_comps(sents_df, word_vecs, start, alpha, beta, gamma, K, f_comp, stopwords)
    return similarity.compute_sims(comps, exp_df, f_sim)


def get_LM_composition_test(sent, tokenizer, model, max_len, mode, device='cpu', start='left'):
    e = functions.tokenize(tokenizer, sent, max_len)

    input_ids = e['input_ids'].flatten().reshape(1, -1).to(device, dtype=torch.long)
    attention_mask = e['attention_mask'].flatten().reshape(1, -1).to(device, dtype=torch.long)
    mask = attention_mask.nonzero()[:, 1]

    # # Create mask to exclude additional sub-tokens and special tokens from the composition
    # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # special_toks = tokenizer.special_tokens_map.values()
    # mask = [i for i, tok in enumerate(tokens) if tok not in special_toks and not tok.startswith('##')]

    # Create copies of the original sentence with each token masked per entry
    skip_model = False
    if not skip_model:
        input_ids = input_ids.repeat(input_ids.shape[1]-1, 1)
        mask_ids = list(range(1, input_ids.shape[1]-1))
        input_ids[mask_ids, mask_ids] = tokenizer.mask_token_id

    # # Obtain the original local embeddings of the input tokens
    # orig = model.get_input_embeddings().weight[input_ids[0, mask]]

    if not skip_model:
        out = model(input_ids=input_ids, attention_mask=attention_mask, mode=mode)[0]
        cls_token = out[:, 0, :]
        if mode == 'CLS':
            out = cls_token.detach().clone()
        else:
            # Third version (Subtracting the CLS token to remove sentence level information - adding it back at the end)
            out = out[:, mask, :] - cls_token
    else:
        if mode == 'CLS':
            out = model.get_input_embeddings().weight[input_ids[0, 0]].unsqueeze(dim=0)
        else:
            out = model.get_input_embeddings().weight[input_ids[0, mask]].unsqueeze(dim=0)

    out = compose_LM(mode, out, None, start)

    if not skip_model and mode != 'CLS':
        out = out + cls_token[0, :]

    return out.detach().cpu().numpy().flatten()


def F_compose_LM(context_embs, global_embs, alpha=1.0, beta=None, gamma=1, K=1, start='left'):
    context_embs = context_embs[0]

    idxs = list(range(context_embs.shape[0]))
    if start == 'right':
        idxs = idxs[::-1]

    if global_embs is not None:
        comp = functions.comp_LM(global_embs[idxs[0]], global_embs[idxs[1]], context_embs[idxs[0]], context_embs[idxs[1]],
                                 alpha, beta, gamma, K)
        idxs = idxs[2:]
    else:
        comp = context_embs[idxs[0]]
        global_embs = context_embs
        idxs = idxs[1:]

    for idx in idxs:
        comp = functions.comp_LM(comp, global_embs[idx], comp, context_embs[idx], alpha, beta, gamma, K)

    return comp


def compute_comps_LM(sents_df, tokenizer, model, max_len, mode, device='cpu', start='left', lm_layer=None):
    comps = sents_df.iloc[:, 0].apply(lambda s: get_LM_composition(s, tokenizer, model, max_len, mode, device,
                                                                   start=start, lm_layer=lm_layer))

    if any(comps.isna()):
        print(len(comps), np.sum(comps.isna()))
        comps = comps.apply(lambda r: r if r is not None else np.zeros((768,)))

    return comps


def get_LM_output(tokenizer, sent, model, max_len, device, mode, lm_layer=None):
    e = functions.tokenize(tokenizer, sent, max_len)

    input_ids = e['input_ids'].flatten().reshape(1, -1).to(device, dtype=torch.long)
    attention_mask = e['attention_mask'].flatten().reshape(1, -1).to(device, dtype=torch.long)
    if isinstance(model, BertModel):
        # The first and last tokens correspond to the CLS and SEP special tokens
        if mode == 'CLS':
            mask = [0]
        else:
            mask = attention_mask.nonzero(as_tuple=False)[1:-1, 1]
    else:
        mask = attention_mask.nonzero(as_tuple=False)[:, 1]

    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    if lm_layer is None:
        out = out['last_hidden_state']
    else:
        # The Language Model contains n hidden layers plus the initial embedding layer [0-12]
        out = out['hidden_states'][lm_layer]
    out = out[:, mask, :]

    return out


def get_LM_composition(sent, tokenizer, model, max_len, mode, device='cpu', start='left', lm_layer=None):
    out = get_LM_output(tokenizer, sent, model, max_len, device, mode, lm_layer)
    out = out.detach().cpu().numpy()
    out = compose_LM(mode, out, None, start)

    return out.flatten()


def compose_LM(mode, out, orig, start):
    if mode == 'SUM':
        # assert(torch.allclose(out.sum(axis=1)[0], F_compose_LM(out, orig, 1, -2, 1, 1, start)))
        out = F_compose_LM(out, orig, 1, -2, 1, 1, start)
    elif mode == 'INF':
        out = F_compose_LM(out, orig, 1, None, 1, 1, start)
    elif mode == 'IND':
        out = F_compose_LM(out, orig, 1, 0, 1, 1, start)
    elif mode == 'JOINT':
        out = F_compose_LM(out, orig, 1, 1, 1, 1, start)
    elif mode == 'AVG':
        out = F_compose_LM(out, orig, 1/4, 1/2, 1, 1, start)
    elif mode == 'GLOBAL_AVG':
        out = out.mean(axis=1)
    elif mode == 'MAX':
        out = out.max(axis=1)

    return out


def sent2sims_LM(sents_df, exp_df, tokenizer, model, max_len, modes, f_sim, device='cpu', start='left', lm_layer=None,
                 verbose=True):
    sims = {}
    for mode in modes:
        if verbose:
            print(f'+++++++ Mode: {mode}')
        comps = compute_comps_LM(sents_df, tokenizer, model, max_len, mode, device, start, lm_layer)
        sims[mode] = similarity.compute_sims(comps, exp_df, f_sim)

    return sims


def sent2sims_SBERT(sents_df, exp_df, model, f_sim):
    comps = pd.Series(list(model.encode(sents_df.text)), index=sents_df.index.values)
    return similarity.compute_sims(comps, exp_df, f_sim)

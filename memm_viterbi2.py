import pickle
from represent_input_with_features2 import represent_input_with_features
import numpy as np


def get_possible_last_tags_lists(pai: dict):
    v = list(pai.values())
    k = list(pai.keys())
    max_key = k[v.index(max(v))]
    v.remove(max(v))
    k.remove(max_key)
    second_max_key = k[v.index(max(v))]
    return [second_max_key[0], max_key[0]], [second_max_key[1], max_key[1]]


def memm_viterbi(all_tags, sentence, weights_path, feature2id):
    """
    Write your MEMM Vitebi imlementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    paramters:
    all_tags - all possible tags
    corpus_words - list of words
    weights_path - path to pickle file with trained weights
    feature2id - the feature2id class
    """
    sentence_words = sentence.split(' ')
    sentence_words.append('')  # ToDo: keep?
    with open(weights_path, 'rb') as f:
        optimal_params = pickle.load(f)
    w_i = optimal_params[0]
    relevant_features_for_idx = dict()
    pai_list = []
    bp_list = []

    # idx = 0
    t = u_tag = '*'
    pai = dict()
    bp = dict()
    for v_tag in all_tags:
        h = (sentence_words[0], t, u_tag, v_tag, sentence_words[1], '*', 0)
        if h not in relevant_features_for_idx:
            relevant_features_for_idx[h] = represent_input_with_features(h, feature2id)

        f_xi_yi = relevant_features_for_idx[h]
        v_mul_f_xi_yi = sum(w_i[f_xi_yi])  # v * f(xi,yi)

        p_numerator = np.exp(v_mul_f_xi_yi)  # exp(v * f(xi,yi))
        p_denominator = 0
        for norm_tag in all_tags:
            h_norm = (sentence_words[0], t, u_tag, norm_tag, sentence_words[1], '*', 0)
            if h_norm not in relevant_features_for_idx:
                relevant_features_for_idx[h_norm] = represent_input_with_features(h_norm, feature2id)

            f_xi_yTag = relevant_features_for_idx[h_norm]
            v_mul_f_xi_yTag = sum(w_i[f_xi_yTag])
            p_denominator += np.exp(v_mul_f_xi_yTag)

        p = p_numerator / p_denominator
        pai[(u_tag, v_tag)] = p
        bp[(u_tag, v_tag)] = '*'
    pai_list.append(pai)
    bp_list.append(bp)
    #

    if len(sentence_words) == 2:
        return get_tag_sequence(pai_list, bp_list)

    # idx = 1
    pai = dict()
    bp = dict()
    t = '*'
    for u_tag in all_tags:
        for v_tag in all_tags:
            h = (sentence_words[1], t, u_tag, v_tag, sentence_words[2], sentence_words[0], 1)
            if h not in relevant_features_for_idx:
                relevant_features_for_idx[h] = represent_input_with_features(h, feature2id)

            f_xi_yi = relevant_features_for_idx[h]
            v_mul_f_xi_yi = sum(w_i[f_xi_yi])  # v * f(xi,yi)

            p_numerator = np.exp(v_mul_f_xi_yi)  # exp(v * f(xi,yi))
            p_denominator = 0
            for norm_tag in all_tags:
                h_norm = (sentence_words[1], t, u_tag, norm_tag, sentence_words[2], sentence_words[0], 1)
                if h_norm not in relevant_features_for_idx:
                    relevant_features_for_idx[h_norm] = represent_input_with_features(h_norm, feature2id)

                f_xi_yTag = relevant_features_for_idx[h_norm]
                v_mul_f_xi_yTag = sum(w_i[f_xi_yTag])
                p_denominator += np.exp(v_mul_f_xi_yTag)

            p = p_numerator / p_denominator
            prev_pai = pai_list[0]
            pai[(u_tag, v_tag)] = p * prev_pai[(t, u_tag)]
            bp[(u_tag, v_tag)] = '*'
    pai_list.append(pai)
    bp_list.append(bp)

    if len(sentence_words) == 3:
        return get_tag_sequence(pai_list, bp_list)
    #

    for idx in range(2, len(sentence_words) - 1):
        pai = dict()
        bp = dict()
        t_list, u_list = get_possible_last_tags_lists(pai_list[idx - 1])
        for u_tag in u_list:
            for v_tag in all_tags:
                prob_lst_for_t = dict()
                for t in t_list:
                    h = (sentence_words[idx], t, u_tag, v_tag, sentence_words[idx + 1], sentence_words[idx - 1], idx)

                    if h not in relevant_features_for_idx:
                        relevant_features_for_idx[h] = represent_input_with_features(h, feature2id)

                    f_xi_yi = relevant_features_for_idx[h]
                    v_mul_f_xi_yi = sum(w_i[f_xi_yi])  # v * f(xi,yi)

                    p_numerator = np.exp(v_mul_f_xi_yi)  # exp(v * f(xi,yi))
                    p_denominator = 0
                    for norm_tag in all_tags:
                        h_norm = (
                        sentence_words[idx], t, u_tag, norm_tag, sentence_words[idx + 1], sentence_words[idx - 1], idx)
                        if h_norm not in relevant_features_for_idx:
                            relevant_features_for_idx[h_norm] = represent_input_with_features(h_norm, feature2id)

                        f_xi_yTag = relevant_features_for_idx[h_norm]
                        v_mul_f_xi_yTag = sum(w_i[f_xi_yTag])
                        p_denominator += np.exp(v_mul_f_xi_yTag)

                    p = p_numerator / p_denominator
                    prev_pai = pai_list[idx - 1]
                    prob_lst_for_t[t] = (p * prev_pai[(t, u_tag)])

                max_t = get_key_with_max_val(prob_lst_for_t)
                pai[(u_tag, v_tag)] = prob_lst_for_t[max_t]
                bp[(u_tag, v_tag)] = max_t

        pai_list.append(pai)
        bp_list.append(bp)

    return get_tag_sequence(pai_list, bp_list)


def get_tag_sequence(pai_list, bp_list):
    tag_sequence = []
    n = len(pai_list) - 1
    pai = pai_list[n]
    (ntag, nntag) = get_key_with_max_val(pai)
    tag_sequence.append(nntag)
    if ntag != '*':
        tag_sequence.append(ntag)

    for i in range(n - 2, -1, -1):
        bp = bp_list[i + 2]
        tk = bp[(ntag, nntag)]
        tag_sequence.append(tk)
        nntag = ntag
        ntag = tk
    tag_sequence.reverse()

    return tag_sequence


def get_key_with_max_val(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

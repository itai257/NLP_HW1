from feature_statistics_class import feature_statistics_class
from feature2id_class import feature2id_class
from represent_input_with_features import represent_input_with_features
from collections import OrderedDict
from scipy.optimize import fmin_l_bfgs_b
from calc_objective_per_iter import calc_objective_per_iter
import numpy as np

threshold = 10

train_path = "../datashare/hw1/train1.wtag"

# Statistics
statistics = feature_statistics_class()
statistics.get_word_tag_pair_count(train_path)

# feature2id
feature2id = feature2id_class(statistics, threshold)

feature2id.get_word_tag_pairs(train_path)
feature2id.get_suffixes_tags(train_path)
feature2id.get_prefixes_tags(train_path)
feature2id.get_feature_103(train_path)
feature2id.get_feature_104(train_path)
feature2id.get_feature_105(train_path)

relevant_features_for_idx = dict()
all_tags = statistics.feature_105_dict.keys()


histories = dict()
with open(train_path) as f:
    for line in f:
        splited_words = line.replace('\n', ' ').split(' ')
        del splited_words[-1]
        pptag = ptag = '*'
        for word_idx in range(len(splited_words)):
            cur_word, cur_tag = splited_words[word_idx].split('_')
            h = (cur_word, pptag, ptag, cur_tag, '', '')
            if h not in histories:
                histories[h] = 1
                relevant_features_for_idx[h] = (represent_input_with_features(
                    h,
                    feature2id.words_tags_dict,
                    feature2id.suffixes_tags_dict,
                    feature2id.prefixes_tags_dict,
                    feature2id.feature_103_dict,
                    feature2id.feature_104_dict,
                    feature2id.feature_105_dict))
            else:
                histories[h] += 1

            pptag = ptag
            ptag = cur_tag

all_tags_histories = dict()
rel_features_for_all_tags_hist = dict()
for hist, reps in histories.items():
    for tag in all_tags:  # TBD: need to find more tags ?
        h = (hist[0], hist[1], hist[2], tag, hist[4], hist[5])
        if h not in all_tags_histories.keys():
            all_tags_histories[h] = reps
            rel_features_for_all_tags_hist[h] = (represent_input_with_features(
                h,
                feature2id.words_tags_dict,
                feature2id.suffixes_tags_dict,
                feature2id.prefixes_tags_dict,
                feature2id.feature_103_dict,
                feature2id.feature_104_dict,
                feature2id.feature_105_dict))
        else:
            all_tags_histories[h] += reps  ##



n_total_features = feature2id.n_total_features
w_0 = np.zeros(n_total_features, dtype=np.float32)
args = (feature2id, histories, relevant_features_for_idx, all_tags_histories, rel_features_for_all_tags_hist)

optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]
print("weights:")
print(weights)

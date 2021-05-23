from feature_statistics_class import feature_statistics_class
from feature2id_class import feature2id_class
from represent_input_with_features import represent_input_with_features
from collections import OrderedDict
from scipy.optimize import fmin_l_bfgs_b
from calc_objective_per_iter import calc_objective_per_iter
import numpy as np
import pickle
import time

class iter_count():
    def __init__(self):
        self.n = 0

threshold = 30
start1 = time.time()
train_path = "/datashare/hw1/train1.wtag"
#train_path = "data/train1.wtag"

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


histories = []
with open(train_path) as f:
    for line in f:
        splited_words = line.replace('\n', ' ').split(' ')
        del splited_words[-1]
        pptag = ptag = '*'
        for word_idx in range(len(splited_words)):
            cur_word, cur_tag = splited_words[word_idx].split('_')
            h = (cur_word, pptag, ptag, cur_tag, '', '')
            histories.append(h)
            if h not in relevant_features_for_idx:
                relevant_features_for_idx[h] = (represent_input_with_features(
                    h,
                    feature2id.words_tags_dict,
                    feature2id.suffixes_tags_dict,
                    feature2id.prefixes_tags_dict,
                    feature2id.feature_103_dict,
                    feature2id.feature_104_dict,
                    feature2id.feature_105_dict))

            pptag = ptag
            ptag = cur_tag

rel_features_for_all_tags_hist = dict()
for hist in histories:
    for tag in all_tags:  # ToDo: need to find more tags ?
        h = (hist[0], hist[1], hist[2], tag, hist[4], hist[5])
        if h not in rel_features_for_all_tags_hist:
            rel_features_for_all_tags_hist[h] = (represent_input_with_features(
                h,
                feature2id.words_tags_dict,
                feature2id.suffixes_tags_dict,
                feature2id.prefixes_tags_dict,
                feature2id.feature_103_dict,
                feature2id.feature_104_dict,
                feature2id.feature_105_dict))

iteration_count = iter_count()
n_total_features = feature2id.n_total_features
w_0 = np.zeros(n_total_features, dtype=np.float64)
args = (feature2id, histories, relevant_features_for_idx, all_tags, rel_features_for_all_tags_hist, iteration_count)

optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]
print("weights:")
print(weights)
end1 = time.time()
print("total time:")
print(end1 - start1)

weights_path = 'trained_weights/trained_weights_data_train1.pkl'
with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

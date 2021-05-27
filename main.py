from feature_statistics_class import feature_statistics_class
from feature2id_class import feature2id_class
from represent_input_with_features import represent_input_with_features
from memm_viterbi import memm_viterbi
from collections import OrderedDict
from scipy.optimize import fmin_l_bfgs_b
from calc_objective_per_iter import calc_objective_per_iter
import numpy as np
import pickle
import time


class iter_count():
    def __init__(self):
        self.n = 0


threshold = 5
lamda = 2.5

start1 = time.time()
# train_path = "/datashare/hw1/train1.wtag"
train_path = "data/train1.wtag"

# Statistics
statistics = feature_statistics_class()
statistics.get_word_tag_pair_count(train_path)

# feature2id
feature2id = feature2id_class(statistics, threshold)
feature2id.get_features()

relevant_features_for_idx = dict()
all_tags = statistics.tags_count_dict.keys()


histories = dict()
unique_hist_count = 0
with open(train_path) as f:
    for line in f:
        splited_words = line.replace('\n', ' ').split(' ')
        del splited_words[-1]
        pptag, ptag = '*', '*'
        for word_idx in range(len(splited_words)):
            cur_word, cur_tag = splited_words[word_idx].split('_')
            h = (cur_word, pptag, ptag, cur_tag, '', '')
            if h not in histories:
                histories[h] = 1
                relevant_features_for_idx[h] = (represent_input_with_features(h, feature2id))
                unique_hist_count += 1
            else:
                histories[h] += 1

            pptag = ptag
            ptag = cur_tag
print(unique_hist_count)
rel_features_for_all_tags_hist = dict()
for hist, reps in histories.items():
    for tag in all_tags:  # ToDo: need to find more tags ?
        h = (hist[0], hist[1], hist[2], tag, hist[4], hist[5])
        if h not in rel_features_for_all_tags_hist:
            rel_features_for_all_tags_hist[h] = (represent_input_with_features(h, feature2id))

weights_path = 'trained_weights/trained_weights_data_train1.pkl'
iteration_count = iter_count()
n_total_features = feature2id.n_total_features
w_0 = np.zeros(n_total_features, dtype=np.float64)
# """
args = (feature2id, histories, relevant_features_for_idx, all_tags, rel_features_for_all_tags_hist, iteration_count, lamda)

optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]

with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

print("weights:")
print(weights)
# """
def get_sentence_and_tags(line):
    words = []
    tags = []
    splited_words = line.replace('\n', ' ').split(' ')
    del splited_words[-1]
    for word_idx in range(len(splited_words)):
        cur_word, cur_tag = splited_words[word_idx].split('_')
        words.append(cur_word)
        tags.append(cur_tag)
    sentence = ' '.join(words)
    return sentence, tags

## Testing:
test_path1 = "/datashare/hw1/test1.wtag"
#test_path1 = "data/test1.wtag"
accuracy_list = []
with open(test_path1) as f:

    for line in f:
        sen, real_tags = get_sentence_and_tags(line)
        infer_tags = memm_viterbi(all_tags, sen, weights_path, feature2id)
        accuracy = (np.count_nonzero(np.array(infer_tags) == np.array(real_tags)) / len(infer_tags)) * 100
        accuracy_list.append(accuracy)
        print("Accuracy: {}".format(accuracy))
print("Test accuracy with train1 data:", )
print(sum(accuracy_list) / len(accuracy_list))
end1 = time.time()
print("total time:")
print(end1 - start1)

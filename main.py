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

threshold = 5
lamda = 2.5

total_time_start = time.time()
pre_process_time_start = time.time()
print("Starting pre-process phase:")
train_path = "/datashare/hw1/train1.wtag"
# train_path = "data/train1.wtag"

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

pre_process_time_end = time.time()
print("end pre process phase, time: {}".format(pre_process_time_end - pre_process_time_start))
print("----")

training_time_start = time.time()
print("Starting training phase:")

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

training_time_end = time.time()
print("end training phase, time: {}".format(training_time_end - training_time_start))
print("----")

inference_time_start = time.time()
print("Starting inference phase:")


## Testing:
test_path1 = "/datashare/hw1/test1.wtag"
#test_path1 = "data/test1.wtag"

tags_infer_mistakes_cnt = dict()
all_tags_real_infer_dict = dict()
accuracy_list = []

for tag in all_tags:
    tags_infer_mistakes_cnt[tag] = 0
    for tag2 in all_tags:
        all_tags_real_infer_dict[(tag, tag2)] = 0

with open(test_path1) as f:

    for line in f:
        sen, real_tags = get_sentence_and_tags(line)
        infer_tags = memm_viterbi(all_tags, sen, weights_path, feature2id)
        true_false_arr = (np.array(infer_tags) == np.array(real_tags))
        accuracy = (np.count_nonzero(true_false_arr) / len(infer_tags)) * 100
        accuracy_list.append(accuracy)
        false_infer_tags = np.array(real_tags)[true_false_arr == False]

        for i in range(len(real_tags)):
            all_tags_real_infer_dict[(real_tags[i], infer_tags[i])] += 1

        for t in false_infer_tags:
            tags_infer_mistakes_cnt[t] += 1

inference_time_end = time.time()
print("end inference phase, time: {}".format(inference_time_end - inference_time_start))
print("----")

max_mistakes_tags = sorted(tags_infer_mistakes_cnt, key=tags_infer_mistakes_cnt.get, reverse=True)[:10]
for t1 in max_mistakes_tags:
    for t2 in max_mistakes_tags:
        val = all_tags_real_infer_dict[(t1,t2)]
        print("real tag: {}, inference tag: {}, value: {}".format(t1, t2, val))

print("Test accuracy with train1 data:")
print(sum(accuracy_list) / len(accuracy_list))
total_time_end = time.time()
print("total time:")
print(total_time_end - total_time_start)
print("iterations count:")
print(iteration_count.n)

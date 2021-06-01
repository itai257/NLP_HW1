from featurestatisticsclass import FeatureStatisticsClass
from feature2idclass import Feature2IdClass
from represent_input_with_features import represent_input_with_features
from calc_objective_per_iter import calc_objective_per_iter
from memm_viterbi import memm_viterbi

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import pickle

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
train_path = "data/train1.wtag"

# Statistics
statistics = FeatureStatisticsClass()
statistics.get_features(train_path)

# feature2id
feature2id = Feature2IdClass(statistics)
feature2id.get_features()

relevant_features_for_idx = dict()
all_tags = statistics.tags_count_dict.keys()


## save feature2id to disk
with open('trained_weights/feature2id_train1.pkl', 'wb+') as output:
    pickle.dump(feature2id, output, pickle.HIGHEST_PROTOCOL)
##

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
# """
n_total_features = feature2id.n_total_features
w_0 = np.zeros(n_total_features, dtype=np.float64)

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


# Testing:
test_path1 = "/datashare/hw1/test1.wtag"
test_path1 = "data/test1.wtag"

tags_infer_mistakes_cnt = dict()
all_tags_real_infer_dict = dict()

for tag in all_tags:
    tags_infer_mistakes_cnt[tag] = 0
    for tag2 in all_tags:
        all_tags_real_infer_dict[(tag, tag2)] = 0

with open(train_path) as f:
    for line in f:
        sen, real_tags = get_sentence_and_tags(line)
        infer_tags = memm_viterbi(all_tags, sen, weights_path, feature2id)
        true_false_arr = (np.array(infer_tags) == np.array(real_tags))
        false_infer_tags = np.array(real_tags)[true_false_arr == False]

        for i in range(len(real_tags)):
            all_tags_real_infer_dict[(real_tags[i], infer_tags[i])] += 1

        for t in false_infer_tags:
            tags_infer_mistakes_cnt[t] += 1

inference_time_end = time.time()
print("end inference phase, time: {}".format(inference_time_end - inference_time_start))
print("----")
max_mistakes_tags = sorted(tags_infer_mistakes_cnt, key=tags_infer_mistakes_cnt.get, reverse=True)[:10]
all_tags_to_iterate = max_mistakes_tags.copy() + list((set(all_tags) - set(max_mistakes_tags)))  # in order to first iterate max_mistakes_tags and then the rest
for infer_tag in max_mistakes_tags:
    for real_tag in all_tags_to_iterate:
        val = all_tags_real_infer_dict[(real_tag, infer_tag)]
        print("inference tag: {}, real tag: {}, value: {}".format(infer_tag, real_tag, val))

true_infer_count = 0
all_inference_count = 0
for tag in all_tags:
    true_infer_count += all_tags_real_infer_dict[(tag, tag)]
    for tag2 in all_tags:
        all_inference_count += all_tags_real_infer_dict[(tag, tag2)]

accuracy = true_infer_count / all_inference_count
accuracy = accuracy * 100

print("Accuracy: {}".format(accuracy))

total_time_end = time.time()
print("total time:")
print(total_time_end - total_time_start)
print("iterations count:")
print(iteration_count.n)

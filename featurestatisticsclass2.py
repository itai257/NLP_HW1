from collections import OrderedDict


def dict_insert(key, dictionary):
    if key not in dictionary:
        dictionary[key] = 1
    else:
        dictionary[key] += 1


class FeatureStatisticsClass:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.suffixes_tags_count_dict = OrderedDict()
        self.prefixes_tags_count_dict = OrderedDict()
        self.tags_tuples_count_dict = OrderedDict()
        self.tags_pairs_count_dict = OrderedDict()
        self.tags_count_dict = OrderedDict()

        self.prev_words_tags_count_dict = OrderedDict()
        self.next_words_tags_count_dict = OrderedDict()

        # New Features
        self.upper_case_tags_count_dict = OrderedDict()
        self.special_char_tags_count_dict = OrderedDict()
        self.numeric_tags_count_dict = OrderedDict()
        self.lengths_tags_count_dict = OrderedDict()
        # self.indexes_tags_count_dict = OrderedDict()

    def get_features(self, file_path):
        with open(file_path) as f:
            for line in f:
                line_words = line.replace('\n', ' ').split(' ')[:-1]
                pp_tag, p_tag, p_word = '*', '*', ''
                for word_idx in range(len(line_words)):
                    cur_word, cur_tag = line_words[word_idx].split('_')
                    dict_insert((cur_word, cur_tag), self.words_tags_count_dict)

                    for suffix_length in range(2, 5):   # Run from 2 to 4
                        if len(cur_word) > suffix_length:
                            cur_suffix = cur_word[-suffix_length:]
                            dict_insert((cur_suffix, cur_tag), self.suffixes_tags_count_dict)
                            cur_prefix = cur_word[:suffix_length]
                            dict_insert((cur_prefix, cur_tag), self.prefixes_tags_count_dict)
                    dict_insert((pp_tag, p_tag, cur_tag), self.tags_tuples_count_dict)
                    dict_insert((p_tag, cur_tag), self.tags_pairs_count_dict)
                    dict_insert(cur_tag, self.tags_count_dict)

                    if word_idx != 0:
                        dict_insert((p_word, cur_tag), self.prev_words_tags_count_dict)
                        dict_insert((cur_word, p_tag), self.next_words_tags_count_dict)

                    # New features
                    if not cur_word.islower():
                        if p_tag != '*':    # Uppercase but not first in sentence.
                            dict_insert(cur_tag, self.upper_case_tags_count_dict)
                    if not cur_word.isalnum():
                        dict_insert(cur_tag, self.special_char_tags_count_dict)
                    if cur_word.isnumeric():
                        dict_insert(cur_tag, self.numeric_tags_count_dict)
                    dict_insert((len(cur_word), cur_tag), self.lengths_tags_count_dict)

                    pp_tag = p_tag
                    p_tag = cur_tag
                    p_word = cur_word

        self.n_total_features = len(self.words_tags_count_dict)
        self.n_total_features += len(self.suffixes_tags_count_dict)
        self.n_total_features += len(self.prefixes_tags_count_dict)
        self.n_total_features += len(self.tags_tuples_count_dict)
        self.n_total_features += len(self.tags_pairs_count_dict)
        self.n_total_features += len(self.tags_count_dict)

        self.n_total_features = len(self.prev_words_tags_count_dict)
        self.n_total_features = len(self.next_words_tags_count_dict)

        self.n_total_features += len(self.upper_case_tags_count_dict)
        self.n_total_features += len(self.special_char_tags_count_dict)
        self.n_total_features += len(self.numeric_tags_count_dict)
        self.n_total_features += len(self.lengths_tags_count_dict)
        # self.n_total_features += len(self.indexes_tags_count_dict)


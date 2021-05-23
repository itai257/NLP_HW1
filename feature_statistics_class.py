import collections
from collections import OrderedDict

class feature_statistics_class():
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.suffixes_tags_count_dict = OrderedDict()
        self.prefixes_tags_count_dict = OrderedDict()
        self.feature_103_dict = OrderedDict()
        self.feature_104_dict = OrderedDict()
        self.feature_105_dict = OrderedDict()

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = line.replace('\n', ' ').split(' ')
                del splited_words[-1]
                pptag, ptag = '*', '*'

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

                    for suffix_length in range(2, 5):   # Run from 2 to 4
                        if len(cur_word) > suffix_length:
                            cur_suffix = cur_word[-suffix_length:]
                            if (cur_suffix, cur_tag) not in self.suffixes_tags_count_dict:
                                self.suffixes_tags_count_dict[(cur_suffix, cur_tag)] = 1
                            else:
                                self.suffixes_tags_count_dict[(cur_suffix, cur_tag)] += 1

                            cur_prefix = cur_word[:suffix_length]
                            if (cur_prefix, cur_tag) not in self.prefixes_tags_count_dict:
                                self.prefixes_tags_count_dict[(cur_prefix, cur_tag)] = 1
                            else:
                                self.prefixes_tags_count_dict[(cur_prefix, cur_tag)] += 1

                    if (pptag, ptag, cur_tag) not in self.feature_103_dict:
                        self.feature_103_dict[(pptag, ptag, cur_tag)] = 1
                    else:
                        self.feature_103_dict[(pptag, ptag, cur_tag)] += 1

                    if (ptag, cur_tag) not in self.feature_104_dict:
                        self.feature_104_dict[(ptag, cur_tag)] = 1
                    else:
                        self.feature_104_dict[(ptag, cur_tag)] += 1

                    if cur_tag not in self.feature_105_dict:
                        self.feature_105_dict[cur_tag] = 1
                    else:
                        self.feature_105_dict[cur_tag] += 1

                    pptag = ptag
                    ptag = cur_tag

        self.n_total_features = len(self.words_tags_count_dict)
        self.n_total_features += len(self.suffixes_tags_count_dict)
        self.n_total_features += len(self.prefixes_tags_count_dict)
        self.n_total_features += len(self.feature_103_dict)
        self.n_total_features += len(self.feature_104_dict)
        self.n_total_features += len(self.feature_105_dict)


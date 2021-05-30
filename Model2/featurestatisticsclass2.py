from collections import OrderedDict


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

        self.capitals_tags_count_dict = OrderedDict()
        self.first_tags_count_dict = OrderedDict()
        self.second_tags_count_dict = OrderedDict()
        self.lengths_tags_count_dict = OrderedDict()

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                line_words = line.replace('\n', ' ').split(' ')
                del line_words[-1]
                pp_tag, p_tag = '*', '*'
                for word_idx in range(len(line_words)):
                    cur_word, cur_tag = line_words[word_idx].split('_')

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

                    if (pp_tag, p_tag, cur_tag) not in self.tags_tuples_count_dict:
                        self.tags_tuples_count_dict[(pp_tag, p_tag, cur_tag)] = 1
                    else:
                        self.tags_tuples_count_dict[(pp_tag, p_tag, cur_tag)] += 1

                    if (p_tag, cur_tag) not in self.tags_pairs_count_dict:
                        self.tags_pairs_count_dict[(p_tag, cur_tag)] = 1
                    else:
                        self.tags_pairs_count_dict[(p_tag, cur_tag)] += 1

                    if cur_tag not in self.tags_count_dict:
                        self.tags_count_dict[cur_tag] = 1
                    else:
                        self.tags_count_dict[cur_tag] += 1

                    # New features
                    if p_tag != '*':
                        if cur_word != cur_word.lower():
                            if cur_tag not in self.capitals_tags_count_dict:
                                self.capitals_tags_count_dict[cur_tag] = 1
                            else:
                                self.capitals_tags_count_dict[cur_tag] += 1

                        if pp_tag == '*':
                            if cur_tag not in self.second_tags_count_dict:
                                self.second_tags_count_dict[cur_tag] = 1
                            else:
                                self.second_tags_count_dict[cur_tag] += 1

                    else:
                        if cur_tag not in self.first_tags_count_dict:
                            self.first_tags_count_dict[cur_tag] = 1
                        else:
                            self.first_tags_count_dict[cur_tag] += 1

                    w_len = len(cur_word)
                    if (w_len, cur_tag) not in self.lengths_tags_count_dict:
                        self.lengths_tags_count_dict[(w_len, cur_tag)] = 1
                    else:
                        self.lengths_tags_count_dict[(w_len, cur_tag)] += 1

                    pp_tag = p_tag
                    p_tag = cur_tag

        self.n_total_features = len(self.words_tags_count_dict)
        self.n_total_features += len(self.suffixes_tags_count_dict)
        self.n_total_features += len(self.prefixes_tags_count_dict)
        self.n_total_features += len(self.tags_tuples_count_dict)
        self.n_total_features += len(self.tags_pairs_count_dict)
        self.n_total_features += len(self.tags_count_dict)

        self.n_total_features += len(self.capitals_tags_count_dict)
        self.n_total_features += len(self.first_tags_count_dict)
        self.n_total_features += len(self.second_tags_count_dict)
        self.n_total_features += len(self.lengths_tags_count_dict)

from collections import OrderedDict
from featurestatisticsclass2 import FeatureStatisticsClass


class Feature2IdClass:

    def __init__(self, feature_statistics: FeatureStatisticsClass):
        self.statistics = feature_statistics  # statistics class, for each feature gives empirical counts

        self.n_total_features = 0  # Total number of features accumulated
        self.n_words_tags = 0  # Number of Word\Tag pairs features
        self.n_suffixes_tags = 0
        self.n_prefixes_tags = 0
        self.n_tags_tuples = 0
        self.n_tags_pairs = 0
        self.n_tags = 0

        self.n_prev_words_tags = 0
        self.n_next_words_tags = 0

        self.n_uppercase_tags = 0
        self.n_special_char_tags = 0
        self.n_numeric_tags = 0
        self.n_lengths_tags = 0

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.suffixes_tags_dict = OrderedDict()
        self.prefixes_tags_dict = OrderedDict()
        self.tags_tuples_dict = OrderedDict()
        self.tags_pairs_dict = OrderedDict()
        self.tags_dict = OrderedDict()

        self.prev_words_tags_dict = OrderedDict()
        self.next_words_tags_dict = OrderedDict()

        self.uppercase_tags_dict = OrderedDict()
        self.special_char_tags_dict = OrderedDict()
        self.numeric_tags_dict = OrderedDict()
        self.lengths_tags_dict = OrderedDict()

    def get_features(self):
        t = 3
        self.n_words_tags = self.set_best_features_index(self.words_tags_dict,
                                                         self.statistics.words_tags_count_dict, t)
        self.n_suffixes_tags = self.set_best_features_index_2(self.suffixes_tags_dict,
                                                              self.statistics.suffixes_tags_count_dict, t)
        self.n_prefixes_tags = self.set_best_features_index_2(self.prefixes_tags_dict,
                                                              self.statistics.prefixes_tags_count_dict, t)
        self.n_tags_tuples = self.set_best_features_index(self.tags_tuples_dict,
                                                          self.statistics.tags_tuples_count_dict, t)
        self.n_tags_pairs = self.set_best_features_index(self.tags_pairs_dict,
                                                         self.statistics.tags_pairs_count_dict, t)
        self.n_tags = self.set_best_features_index(self.tags_dict,
                                                   self.statistics.tags_count_dict, t)
        print("word tag pair features: {} out of {}".format(self.n_words_tags,
                                                            len(self.statistics.words_tags_count_dict)))
        print("suffix tag pair features:{} out of {}".format(self.n_suffixes_tags,
                                                             len(self.statistics.suffixes_tags_count_dict)))
        print("prefix tag pair features:{} out of {}".format(self.n_prefixes_tags,
                                                             len(self.statistics.prefixes_tags_count_dict)))
        print("tag tuple features:{} out of {}".format(self.n_tags_tuples,
                                                       len(self.statistics.tags_tuples_count_dict)))
        print("tag pair features:{} out of {}".format(self.n_tags_pairs,
                                                      len(self.statistics.tags_pairs_count_dict)))
        print("tag features:{} out of {}".format(self.n_tags,
                                                 len(self.statistics.tags_count_dict)))

        self.n_prev_words_tags = self.set_best_features_index(self.prev_words_tags_dict,
                                                              self.statistics.prev_words_tags_count_dict, t)
        self.n_next_words_tags = self.set_best_features_index(self.next_words_tags_dict,
                                                              self.statistics.next_words_tags_count_dict, t)
        print("previous word tag pair features: {} out of {}".format(self.n_prev_words_tags,
                                                                     len(self.statistics.prev_words_tags_count_dict)))
        print("next word tag pair features: {} out of {}".format(self.n_next_words_tags,
                                                                 len(self.statistics.next_words_tags_count_dict)))

        self.n_uppercase_tags = self.set_best_features_index(self.uppercase_tags_dict,
                                                             self.statistics.upper_case_tags_count_dict, t)
        self.n_special_char_tags = self.set_best_features_index(self.special_char_tags_dict,
                                                                self.statistics.special_char_tags_count_dict, t)
        self.n_numeric_tags = self.set_best_features_index(self.numeric_tags_dict,
                                                           self.statistics.numeric_tags_count_dict, t)
        self.n_lengths_tags = self.set_best_features_index(self.lengths_tags_dict,
                                                           self.statistics.lengths_tags_count_dict, t)
        print("has-uppercase tag pair features: {} out of {}".format(self.n_uppercase_tags,
                                                                     len(self.statistics.upper_case_tags_count_dict)))
        print("has-special tag features: {} out of {}".format(self.n_special_char_tags,
                                                              len(self.statistics.special_char_tags_count_dict)))
        print("is-numeric tag features: {} out of {}".format(self.n_numeric_tags,
                                                             len(self.statistics.numeric_tags_count_dict)))
        print("length tag features: {} out of {}".format(self.n_lengths_tags,
                                                         len(self.statistics.lengths_tags_count_dict)))

        print("total features: {} out of {}".format(self.n_total_features, self.statistics.n_total_features))

    def set_best_features_index(self, f_dict, feature_statistics_dict, threshold):
        index = self.n_total_features
        for key in feature_statistics_dict:
            if key not in f_dict and (feature_statistics_dict[key] >= threshold):
                f_dict[key] = index
                index += 1

        features_count = index - self.n_total_features
        self.n_total_features = index
        return features_count

    def set_best_features_index_2(self, f_dict, feature_statistics_dict, base_threshold):
        index = self.n_total_features
        for key in feature_statistics_dict:
            threshold = base_threshold * (5 - len(key[1]))
            if key not in f_dict and (feature_statistics_dict[key] >= threshold):
                f_dict[key] = index
                index += 1

        features_count = index - self.n_total_features
        self.n_total_features = index
        return features_count

from collections import OrderedDict
from featurestatisticsclass import FeatureStatisticsClass


class Feature2IdClass:

    def __init__(self, feature_statistics: FeatureStatisticsClass):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts

        self.n_total_features = 0  # Total number of features accumulated
        self.n_words_tags = 0  # Number of Word\Tag pairs features
        self.n_suffixes_tags = 0
        self.n_prefixes_tags = 0
        self.n_tags_tuples = 0
        self.n_tags_pairs = 0
        self.n_tags = 0

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

        self.uppercase_tags_dict = OrderedDict()
        self.special_char_tags_dict = OrderedDict()
        self.numeric_tags_dict = OrderedDict()
        self.lengths_tags_dict = OrderedDict()

    def get_features(self):
        t = 5
        self.n_words_tags = self.set_best_features_index(self.words_tags_dict,
                                                         self.feature_statistics.words_tags_count_dict, t)
        self.n_suffixes_tags = self.set_best_features_index_2(self.suffixes_tags_dict,
                                                              self.feature_statistics.suffixes_tags_count_dict, t)
        self.n_prefixes_tags = self.set_best_features_index_2(self.prefixes_tags_dict,
                                                              self.feature_statistics.prefixes_tags_count_dict, t)
        self.n_tags_tuples = self.set_best_features_index(self.tags_tuples_dict,
                                                          self.feature_statistics.tags_tuples_count_dict, t)
        self.n_tags_pairs = self.set_best_features_index(self.tags_pairs_dict,
                                                         self.feature_statistics.tags_pairs_count_dict, t)
        self.n_tags = self.set_best_features_index(self.tags_dict,
                                                   self.feature_statistics.tags_count_dict, t)
        print("word tag pair features:", self.n_words_tags)
        print("suffix tag pair features:", self.n_suffixes_tags)
        print("prefix tag pair features:", self.n_prefixes_tags)
        print("tag tuples features:", self.n_tags_tuples)
        print("tag pair features:", self.n_tags_pairs)
        print("tag features:", self.n_tags)

        self.n_uppercase_tags = self.set_best_features_index(self.uppercase_tags_dict,
                                                             self.feature_statistics.upper_case_tags_count_dict, t)
        self.n_special_char_tags = self.set_best_features_index(self.special_char_tags_dict,
                                                                self.feature_statistics.special_char_tags_count_dict, t)
        self.n_numeric_tags = self.set_best_features_index(self.numeric_tags_dict,
                                                           self.feature_statistics.numeric_tags_count_dict, t)
        self.n_lengths_tags = self.set_best_features_index(self.lengths_tags_dict,
                                                           self.feature_statistics.lengths_tags_count_dict, t)
        print("has-uppercase tag pair features:", self.n_uppercase_tags)
        print("has-special tag features:", self.n_special_char_tags)
        print("is-numeric tag features:", self.n_numeric_tags)
        print("length tag features:", self.n_lengths_tags)

        print("total features:", self.n_total_features)

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

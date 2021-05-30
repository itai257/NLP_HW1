from collections import OrderedDict


class Feature2IdClass:

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_words_tags = 0  # Number of Word\Tag pairs features
        self.n_suffixes_tags = 0
        self.n_prefixes_tags = 0
        self.n_tags_tuples = 0
        self.n_tags_pairs = 0
        self.n_tags = 0

        self.n_capitals_tags = 0
        self.n_first_tags = 0
        self.n_second_tags = 0
        self.n_lengths_tags = 0

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.suffixes_tags_dict = OrderedDict()
        self.prefixes_tags_dict = OrderedDict()
        self.tags_tuples_dict = OrderedDict()
        self.tags_pairs_dict = OrderedDict()
        self.tags_dict = OrderedDict()

        self.capitals_tags_dict = OrderedDict()
        self.first_tags_dict = OrderedDict()
        self.second_tags_dict = OrderedDict()
        self.lengths_tags_dict = OrderedDict()

    def get_features(self):
        t = 5
        self.n_words_tags = self.set_best_features_index(self.words_tags_dict,
                                                         self.feature_statistics.words_tags_count_dict, t)
        print("word tag pair features:", self.n_words_tags)
        self.n_suffixes_tags = self.set_best_features_index_2(self.suffixes_tags_dict,
                                                              self.feature_statistics.suffixes_tags_count_dict, t)
        print("suffix tag pair features:", self.n_suffixes_tags)
        self.n_prefixes_tags = self.set_best_features_index_2(self.prefixes_tags_dict,
                                                              self.feature_statistics.prefixes_tags_count_dict, t)
        print("prefix tag pair features:", self.n_prefixes_tags)
        self.n_tags_tuples = self.set_best_features_index(self.tags_tuples_dict,
                                                          self.feature_statistics.tags_tuples_count_dict, t)
        print("tag tuples features:", self.n_tags_tuples)
        self.n_tags_pairs = self.set_best_features_index(self.tags_pairs_dict,
                                                         self.feature_statistics.tags_pairs_count_dict, t)
        print("tag pair features:", self.n_tags_pairs)

        self.n_tags = self.set_best_features_index(self.tags_dict,
                                                   self.feature_statistics.tags_count_dict, t)
        print("tag features:", self.n_tags)

        self.n_capitals_tags = self.set_best_features_index(self.capitals_tags_dict,
                                                            self.feature_statistics.capitals_tags_count_dict, t)
        print("has-capital tag pair features:", self.n_capitals_tags)
        self.n_first_tags = self.set_best_features_index(self.first_tags_dict,
                                                         self.feature_statistics.first_tags_count_dict, t)
        print("first tag features:", self.n_first_tags)
        self.n_second_tags = self.set_best_features_index(self.second_tags_dict,
                                                          self.feature_statistics.second_tags_count_dict, t)
        print("second tag features:", self.n_second_tags)
        self.n_lengths_tags = self.set_best_features_index(self.lengths_tags_dict,
                                                           self.feature_statistics.lengths_tags_count_dict, t)
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
            threshold = base_threshold * (6 - len(key[1]))
            if key not in f_dict and (feature_statistics_dict[key] >= threshold):
                f_dict[key] = index
                index += 1

        features_count = index - self.n_total_features
        self.n_total_features = index
        return features_count

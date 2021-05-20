from collections import OrderedDict

class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features
        self.n_suffixes_tags = 0
        self.n_prefixes_tags = 0
        self.n_feature_103 = 0
        self.n_feature_104 = 0
        self.n_feature_105 = 0

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.suffixes_tags_dict = OrderedDict()
        self.prefixes_tags_dict = OrderedDict()
        self.feature_103_dict = OrderedDict()
        self.feature_104_dict = OrderedDict()
        self.feature_105_dict = OrderedDict()


    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        self.n_tag_pairs = self.set_best_features_index(self.words_tags_dict,
                                                        self.feature_statistics.words_tags_count_dict)


    def get_suffixes_tags(self, file_path):
        self.n_suffixes_tags = self.set_best_features_index(self.suffixes_tags_dict,
                                                            self.feature_statistics.suffixes_tags_count_dict)


    def get_prefixes_tags(self, file_path):
        self.n_prefixes_tags = self.set_best_features_index(self.prefixes_tags_dict,
                                                            self.feature_statistics.prefixes_tags_count_dict)

    def get_feature_103(self, file_path):
        self.n_feature_103 = self.set_best_features_index(self.feature_103_dict,
                                                          self.feature_statistics.feature_103_dict)

    def get_feature_104(self, file_path):
        self.n_feature_104 = self.set_best_features_index(self.feature_104_dict,
                                                          self.feature_statistics.feature_104_dict)

    def get_feature_105(self, file_path):
        self.n_feature_105 = self.set_best_features_index(self.feature_105_dict,
                                                          self.feature_statistics.feature_105_dict)



    def set_best_features_index(self, dict, feature_statistics_dict):
        index = self.n_total_features
        for key in feature_statistics_dict:
            if key not in dict and (feature_statistics_dict[key] >= self.threshold):
                dict[key] = index
                index += 1

        features_count = index - self.n_total_features
        self.n_total_features = index
        return features_count
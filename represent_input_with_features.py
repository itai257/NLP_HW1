import collections
from collections import OrderedDict

def represent_input_with_features(history,
                                  word_tags_dict,
                                  suffix_tags_dict,
                                  prefixes_tags_dict,
                                  feature_103_dict,
                                  feature_104_dict,
                                  feature_105_dict):
    """
        Extract feature vector in per a given history
        :param history: touple{word, pptag, ptag, ctag, nword, pword}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history

            Features_List =
                {(word,tag): index, (word,tag): index}
                feature_vector(x,y) = (g1(x,y), f100(x,y), f101(x,y), f102(x,y), ...)
                g1(x,y) = x is base & y is Vt
    """
    word = history[0]
    pptag = history[1]
    ptag = history[2]
    ctag = history[3]
    nword = history[4]
    pword = history[5]
    features = []

    if (word, ctag) in word_tags_dict:
        features.append(word_tags_dict[(word, ctag)])

    for i in range(1, 5):
        suffix = word[-i:]
        if len(word) > len(suffix):
            if (suffix, ctag) in suffix_tags_dict:
                features.append(suffix_tags_dict[(suffix, ctag)])
        prefix = word[:i]
        if (prefix, ctag) in prefixes_tags_dict:
            features.append(prefixes_tags_dict[(prefix, ctag)])

    if (pptag, ptag, ctag) in feature_103_dict:
        features.append(feature_103_dict[(pptag, ptag, ctag)])

    if (ptag, ctag) in feature_104_dict:
        features.append(feature_104_dict[(ptag, ctag)])

    if ctag in feature_105_dict:
        features.append(feature_105_dict[ctag])

    return features

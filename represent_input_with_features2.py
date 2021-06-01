from feature2idclass2 import Feature2IdClass

def represent_input_with_features(history, features2id: Feature2IdClass):
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

    if (word, ctag) in features2id.words_tags_dict:
        features.append(features2id.words_tags_dict[(word, ctag)])

    for i in range(2, 5):
        suffix = word[-i:]
        if len(word) > len(suffix):
            if (suffix, ctag) in features2id.suffixes_tags_dict:
                features.append(features2id.suffixes_tags_dict[(suffix, ctag)])
        prefix = word[:i]
        if (prefix, ctag) in features2id.prefixes_tags_dict:
            features.append(features2id.prefixes_tags_dict[(prefix, ctag)])

    if (pptag, ptag, ctag) in features2id.tags_tuples_dict:
        features.append(features2id.tags_tuples_dict[(pptag, ptag, ctag)])

    if (ptag, ctag) in features2id.tags_pairs_dict:
        features.append(features2id.tags_pairs_dict[(ptag, ctag)])

    if ctag in features2id.tags_dict:
        features.append(features2id.tags_dict[ctag])

    if ptag != '*' and not word.islower() and ctag in features2id.uppercase_tags_dict:
        features.append(features2id.uppercase_tags_dict[ctag])

    if not word.isalnum() and ctag in features2id.special_char_tags_dict:
        features.append(features2id.special_char_tags_dict[ctag])

    if word.isnumeric() and ctag in features2id.numeric_tags_dict:
        features.append(features2id.numeric_tags_dict[ctag])

    w_len = len(word)
    if (w_len, ctag) in features2id.lengths_tags_dict:
        features.append(features2id.lengths_tags_dict[(w_len, ctag)])
    return features

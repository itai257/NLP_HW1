from feature2idclass import Feature2IdClass

def represent_input_with_features(history, features2id: Feature2IdClass):
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
        if len(word) > i:
            suffix = word[-i:]
            prefix = word[:i]
            if (suffix, ctag) in features2id.suffixes_tags_dict:
                features.append(features2id.suffixes_tags_dict[(suffix, ctag)])
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

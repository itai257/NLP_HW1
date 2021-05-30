from memm_viterbi2 import memm_viterbi
import pickle


def generate_comp_tagged(weights_path, comp_path, generated_file_path, feature2id_class_path):

    with open(feature2id_class_path, 'rb') as input:
        feature2id = pickle.load(input)


    all_tags = feature2id.feature_statistics.tags_count_dict.keys()

    words_tags_to_generate = []
    with open(comp_path) as f:
        for line in f:
            splited_words = line.replace('\n', ' ').split(' ')
            del splited_words[-1]
            infer_tags = memm_viterbi(all_tags, line, weights_path, feature2id)
            words_tags_to_generate.append((splited_words, infer_tags))

    sentences = []
    for line in words_tags_to_generate:
        sentence_parts = []
        for i in range(len(line[0])):
            word, tag = line[0][i], line[1][i]
            word_tag = "_".join([word, tag])
            sentence_parts.append(word_tag)
        sentence = " ".join(sentence_parts)
        sentences.append(sentence)

    all_text = "\n".join(sentences)
    with open(generated_file_path, "w+") as f:
        f.write(all_text)


    # validating generated file

    with open(generated_file_path) as f:
        with open(comp_path) as f_comp:
            for line, line_comp in zip(f, f_comp):
                splited_words = line.replace('\n', ' ').split(' ')
                del splited_words[-1]
                comp_words = line_comp.replace('\n', ' ').split(' ')
                del comp_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if cur_word != comp_words[word_idx]:
                        print("files are different")
                        print("generated: {}".format(line))
                        print("real: {}".format(line_comp))
                        exit(1)
    print("generated file is valid")


weights_path = 'trained_weights2/trained_weights_data_train2.pkl'
# comp_path = "data/comp2.words"
comp_path = "/datashare/hw1/comp2.words"
generated_file_path = "generated_comp/comp2.wtag"   # where to dump the tagged file
feature2id_class_path = 'trained_weights/feature2id_train2.pkl'

generate_comp_tagged(weights_path, comp_path, generated_file_path, feature2id_class_path)

import os
import sys
from datetime import datetime

import numpy as np

from settings import *
from utils import make_dict, json_load, write_lines


def relative_distance(i, e_start, e_end):
    if i < e_start:
        dis = i - e_start
    elif e_start <= i <= e_end:
        dis = 0
    else:
        dis = i - e_end

    if dis < MIN_DISTANCE:
        dis = MIN_DISTANCE
    if dis > MAX_DISTANCE:
        dis = MAX_DISTANCE
    return str(dis)


class Sentence:
    def __init__(self, e1start, e1end, e2start, e2end, words):
        self.e1start = e1start
        self.e1end = e1end
        self.e2start = e2start
        self.e2end = e2end
        self.words = words

        self.positions_1 = []
        self.positions_2 = []
        self.words_encoded = np.zeros(SEQUENCE_LEN, dtype='int32')
        self.e1 = None
        self.e2 = None
        self.e1_context = None
        self.e2_context = None

        self.segments = np.zeros([SEQUENCE_LEN, 3])
        for i in range(min(SEQUENCE_LEN, len(words))):
            if i < e1start:
                self.segments[i, 0] = 1.
            elif e1start <= i <= e2end:
                self.segments[i, 1] = 1.
            elif e2end < i:
                self.segments[i, 2] = 1.

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return "{} {} {} {} {}".format(self.e1start, self.e1end, self.e2start, self.e2end, " ".join(self.words))

    def generate_features(self, encoder):
        for i in range(min(len(self.words), SEQUENCE_LEN)):
            w = self.words[i]
            self.words_encoded[i] = encoder.word_vec(w)

        for i in range(SEQUENCE_LEN):
            self.positions_1.append(encoder.dis1_vec(i, self.e1start, self.e1end))
            self.positions_2.append(encoder.dis2_vec(i, self.e2start, self.e2end))

        self.e1, self.e1_context = self.entity_context(self.e1start, self.e1end, encoder)
        self.e2, self.e2_context = self.entity_context(self.e2start, self.e2end, encoder)

        return self.words_encoded, self.positions_1, self.positions_2, self.e1, self.e2, self.e1_context, self.e2_context, self.segments

    def entity_context(self, e_start, e_end, encoder):
        entity = np.zeros(ENTITY_LEN)
        for i in range(e_end - e_start + 1):
            entity[i] = encoder.word_vec(self.words[e_start + i])

        context = np.zeros(2)
        if e_start > 0:
            context[0] = encoder.word_vec(self.words[e_start - 1])
        if e_end < len(self.words) - 1:
            context[1] = encoder.word_vec(self.words[e_end + 1])
        return entity, context


def read_input(path):
    sentences = []
    y = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            relation, e1start, e1end, e2start, e2end, *words = line.strip().split()
            y.append(int(relation))
            s = Sentence(int(e1start), int(e1end), int(e2start), int(e2end), words)
            sentences.append(s)
    return sentences, y


def read_relations(path):
    idx2relation = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            idx, relation = line.strip().split()
            idx2relation[int(idx)] = relation
    return idx2relation


class Encoder:
    def __init__(self, word2idx, dis2idx_1, dis2idx_2):
        self.word2idx = word2idx
        self.dis2idx_1 = dis2idx_1
        self.dis2idx_2 = dis2idx_2

    def word_vec(self, w):
        if w in self.word2idx:
            return self.word2idx[w]
        else:
            return self.word2idx["UNKNOWN"]

    def dis1_vec(self, i, e_start, e_end):
        d = relative_distance(i, e_start, e_end)
        return self.dis2idx_1[d]

    def dis2_vec(self, i, e_start, e_end):
        d = relative_distance(i, e_start, e_end)
        return self.dis2idx_2[d]


def main(*args):
    assert len(args) >= 2

    word_embeddings = np.load("embedding/word_embeddings.npy")
    position_embeddings_1 = np.load("embedding/position_embeddings_1.npy")
    position_embeddings_2 = np.load("embedding/position_embeddings_2.npy")
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2)

    from models import build_model
    model = build_model(embeddings)
    weights_path = args[0]
    model.load_weights(weights_path)

    dis2idx_1 = json_load("embedding/dis2idx_1.json")
    dis2idx_2 = json_load("embedding/dis2idx_2.json")
    word2idx = json_load("embedding/word2idx.json")
    encoder = Encoder(word2idx, dis2idx_1, dis2idx_2)

    input_file = args[1]
    sentences, y = read_input(input_file)
    data = list(map(list, zip(*[s.generate_features(encoder) for s in sentences])))

    scores = model.predict(data, verbose=False)
    predictions = scores.argmax(-1)
    idx2relation = read_relations("origin_data/relations.txt")
    outputs = ["{} {}".format(prediction, idx2relation[prediction]) for prediction in predictions]

    print("\n".join(outputs))

    timestamp = int(datetime.now().timestamp())
    output_folder = "output/test/%d" % timestamp
    os.makedirs(output_folder, exist_ok=True)
    print("output folder: %s" % output_folder)
    output_file = os.path.join(output_folder, 'output.txt')
    error_list_file = os.path.join(output_folder, 'error_list.txt')
    error_predictions_file = os.path.join(output_folder, 'error_predictions.txt')

    write_lines(output_file, outputs)

    error_list = []
    error_predictions = []
    for sentence, label, prediction in zip(sentences, y, predictions):
        if label != prediction:
            error_list.append('{} {}'.format(label, str(sentence)))
            error_predictions.append('{} {}'.format(prediction, idx2relation[prediction]))

    write_lines(error_list_file, error_list)
    write_lines(error_predictions_file, error_predictions)


if __name__ == '__main__':
    main(*sys.argv[1:])

import json
from collections import OrderedDict

import numpy as np
from settings import *
from utils import make_dict
from os import makedirs
from os.path import exists
from gensim.models import Word2Vec
import pickle
import codecs


class Counter:
    def __init__(self):
        self.max_sequence_len = 0
        self.max_entity_len = 0
        self.vocab_word = set()
        self.distances_1 = []
        self.distances_2 = []
        self.max_entity_distance = 0

    def update(self, sentence):
        self.max_sequence_len = max(self.max_sequence_len, len(sentence))
        self.max_entity_len = max(self.max_entity_len, sentence.max_entity_len)
        self.vocab_word = self.vocab_word | sentence.vocab_word
        self.distances_1.append(sentence.distances_1)
        self.distances_2.append(sentence.distances_2)
        self.max_entity_distance = max(self.max_entity_distance, sentence.e2end - sentence.e1start)

    def __str__(self):
        return "max_sequence_len = %d, max_entity_len = %d, max_entity_distance = %d, vocab_word = %d" % (self.max_sequence_len, self.max_entity_len, self.max_entity_distance, len(self.vocab_word))


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

        self.max_entity_len = max(self.e1end - self.e1start + 1, self.e2end - self.e2start + 1)

        self.vocab_word = set(words)

        self.distances_1 = []
        self.distances_2 = []
        for i in range(len(words)):
            self.distances_1.append(relative_distance(i, e1start, e1end))
            self.distances_2.append(relative_distance(i, e2start, e2end))

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


def read_file(path, counter):
    sentences = []
    y = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            r, e1start, e1end, e2start, e2end, *words = line.strip().split()
            y.append(int(r))
            s = Sentence(int(e1start), int(e1end), int(e2start), int(e2end), words)
            counter.update(s)
            sentences.append(s)
    return sentences, y


def read_word_embeddings(vocab):
    unk_word = np.load("origin_data/unknown.npy")
    word2idx = {
        "PADDING": 0,
        "UNKNOWN": 1,
    }
    word_embeddings = [
        np.zeros(WORD_EMBED_SIZE),
        unk_word
    ]
    vectors = np.load("origin_data/vectors.npy")
    words = json.load(open("origin_data/words.json", "r", encoding="utf8"))
    for w, values in zip(words, vectors):
        word2idx[w] = len(word2idx)
        word_embeddings.append(values)
    unk_words = [w for w in vocab if w not in words]
    json.dump(unk_words, open("data/embedding/unk_words.json", "w", encoding="utf8"), ensure_ascii=False)
    json.dump(word2idx, open("data/embedding/word2idx.json", "w", encoding="utf8"), ensure_ascii=False)
    np.save("data/embedding/word_embeddings.npy", word_embeddings)
    return word2idx


class Encoder:
    def __init__(self, word2idx, dis2idx_1, dis2idx_2):
        self.word2idx = word2idx
        self.dis2idx_1 = dis2idx_1
        self.dis2idx_2 = dis2idx_2
        self.unknown_words = set()

    def word_vec(self, w):
        if w in self.word2idx:
            return self.word2idx[w]
        else:
            self.unknown_words.add(w)
            return self.word2idx["UNKNOWN"]

    def dis1_vec(self, i, e_start, e_end):
        d = relative_distance(i, e_start, e_end)
        return self.dis2idx_1[d]

    def dis2_vec(self, i, e_start, e_end):
        d = relative_distance(i, e_start, e_end)
        return self.dis2idx_2[d]

    def __str__(self):
        return "unknown_words: %d" % len(self.unknown_words)


def numpy_save_many(_dict_):
    for k, data in _dict_.items():
        np.save("data/%s.npy" % k, data)


def pretrain_embedding(data, size, padding=False, unknown=None):
    model = Word2Vec(data, size=size, min_count=1)
    model.init_sims(replace=True)
    index = {}
    embeddings = []
    if padding:
        index["PADDING"] = len(index)
        embeddings.append(np.zeros(size))
    if unknown is not None:
        index["UNKNOWN"] = len(index)
        embeddings.append(unknown)
    for d in model.wv.index2word:
        index[d] = len(index)
        embeddings.append(model.wv.word_vec(d))
    return index, embeddings


def main():
    for folder in ["data/embedding"]:
        makedirs(folder, exist_ok=True)

    counter = Counter()

    print("read data")
    sentences, y = read_file("origin_data/data.cln", counter)

    print(counter)

    print("read word embeddings")
    word2idx = read_word_embeddings(counter.vocab_word)

    print("load position embeddings")
    dis2idx_1, position_embeddings_1 = pretrain_embedding(counter.distances_1, POSITION_EMBED_SIZE)
    json.dump(dis2idx_1, open("data/embedding/dis2idx_1.json", "w"))
    np.save("data/embedding/position_embeddings_1.npy", position_embeddings_1)
    dis2idx_2, position_embeddings_2 = pretrain_embedding(counter.distances_2, POSITION_EMBED_SIZE)
    json.dump(dis2idx_2, open("data/embedding/dis2idx_2.json", "w"))
    np.save("data/embedding/position_embeddings_2.npy", position_embeddings_2)

    encoder = Encoder(word2idx, dis2idx_1, dis2idx_2)

    print("saving data")
    words, pos1, pos2, e1, e2, e1context, e2context, segments = zip(*[s.generate_features(encoder) for s in sentences])
    data = make_dict(words, pos1, pos2, e1, e2, e1context, e2context, segments, y)
    numpy_save_many(data)

    print(encoder)


if __name__ == "__main__":
    main()

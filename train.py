import json
import os
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from metrics import F1score
from models import build_model
from settings import *
from utils import make_dict
from sklearn.model_selection import StratifiedKFold
from metrics import evaluate
from datetime import datetime


def train(split, x, y, x_index, embeddings, log_dir):
    f1_scores = []
    for i, (train_index, test_index) in enumerate(split):
        fold_dir = "%s/fold_%d" % (log_dir, i + 1)
        os.makedirs(fold_dir, exist_ok=True)
        print("training fold %d" % (i + 1))
        weights_path = "%s/weights.best.h5" % fold_dir

        np.save("%s/train_index.npy" % fold_dir, train_index)
        np.save("%s/test_index.npy" % fold_dir, test_index)

        callbacks = [
            TensorBoard(fold_dir),
            F1score(),
            ModelCheckpoint(weights_path, monitor='f1', verbose=1, save_best_only=True, save_weights_only=True,
                            mode='max'),
            EarlyStopping(patience=5, monitor='f1', mode='max')
        ]

        x_train = [d[x_index[train_index]] for d in x]
        y_train = y[train_index]
        x_test = [d[x_index[test_index]] for d in x]
        y_test = y[test_index]
        model = build_model(embeddings)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, verbose=2, callbacks=callbacks,
                  validation_data=[x_test, y_test])

        print("testing fold %d" % (i + 1))
        model.load_weights(weights_path)
        scores = model.predict(x_test, verbose=False)
        predictions = scores.argmax(-1)
        f1 = evaluate(y_test, predictions, "%s/result.json" % fold_dir)
        print("f1_score: %.2f" % f1)
        f1_scores.append(f1)
    f1_avg = np.average(f1_scores)
    max_f1 = max(f1_scores)
    best_fold = int(np.argmax(f1_scores)) + 1
    best_weights = "%s/fold_%d/weights.best.h5" % (log_dir, best_fold)
    result = make_dict(f1_avg, max_f1, best_fold, best_weights)
    print(result)


def main():
    print("load data")
    x = [np.load("data/%s.npy" % name) for name in ["words", "pos1", "pos2", "e1", "e2", "e1context", "e2context", "segments"]]
    y = np.load("data/y.npy")
    x_index = np.arange(len(y))
    skf = StratifiedKFold(n_splits=K_FOLD)

    print("load embeddings")
    word_embeddings = np.load("data/embedding/word_embeddings.npy")
    position_embeddings_1 = np.load("data/embedding/position_embeddings_1.npy")
    position_embeddings_2 = np.load("data/embedding/position_embeddings_2.npy")
    embeddings = make_dict(word_embeddings, position_embeddings_1, position_embeddings_2)

    print("training")
    config = K.tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess = K.tf.Session(config=config)
    K.set_session(sess)

    timestamp = int(datetime.now().timestamp())
    log_dir = "output/train/%d" % timestamp
    print("log_dir = %s" % log_dir)
    split = skf.split(x_index, y)
    split = list(split)

    log_result = train(split, x, y, x_index, embeddings, log_dir)
    json.dump(log_result, open("%s/result.json" % log_dir, "w", encoding="utf8"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

import numpy as np
import keras
import tensorflow as tf
import os
import sys

sys.path.append('../')
sys.path.append('/root/')  # for docker
from loglizer import preprocessing
from loglizer.models import lstm_attention_count_vector
from workflow.BGL_workflow.data_generator import load_BGL
from workflow import dataloader
from scripts import config
from collector.collector import Collector

flags = tf.app.flags
flags.DEFINE_integer('epochs', 100, 'epochs to train')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('g', 5, 'the cutoff in the prediction output to be considered normal')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('L', 2, 'number of layers')
flags.DEFINE_integer('alpha', 64, 'number of memory units')
flags.DEFINE_integer('plb', 11,
                     'padding lower bound, pad to this amount')  # this should be set to prevent length of block < window size, wipe those block with length < window size if this amount is set to 0
flags.DEFINE_string('checkpoint_name', 'lstm_attention_plus_count_vector_unsupervised.h5', 'training directory')
flags.DEFINE_string('result_folder', 'result', 'folder to save results')
flags.DEFINE_float('max_mismatch_rate', 0, 'max rate of mismatch tolerated')
flags.DEFINE_integer('no_repeat_series', 1, 'whether series will not be repeated: 1: no repeat; 0: repeat')
flags.DEFINE_integer('checkpoint_frequency', 10, 'every ? epochs to save checkpoints')
FLAGS = flags.FLAGS


def get_batch_count(a, batch_size):
    return (len(a) - 1) // batch_size + 1


def gen_batch(batch_size, i, cv, l=None, shuffle=False):
    while True:
        if shuffle:
            randnum = np.random.randint(0, 10000)
            np.random.seed(randnum)
            np.random.shuffle(i)
            np.random.seed(randnum)
            np.random.shuffle(cv)
            if l is not None:
                np.random.seed(randnum)
                np.random.shuffle(l)

        c = get_batch_count(i, batch_size)
        for k in range(0, c * batch_size, batch_size):
            u = len(i) if k + batch_size > len(i) else k + batch_size
            if l is not None:
                yield [i[k: u], cv[k: u]], l[k: u]
            else:
                yield [i[k: u], cv[k: u]]


def compare(output, target):
    mismatch = 0
    for i in range(len(target)):
        t = np.sum(output[i] * target[i])
        s = list(output[i])
        s = sorted(s, reverse=True)
        found = False
        for j in range(FLAGS.g):
            if s[j] == t:
                found = True
                break
        if not found:
            mismatch += 1
    mismatch_rate = mismatch / len(target)
    if mismatch_rate > FLAGS.max_mismatch_rate:
        return 1  # anomaly
    else:
        return 0  # normal


def apply_model(x, y, model, mode='inference', collector=None):
    print('== Start %s ==' % mode)
    print('== Generate %s inputs ==' % mode)
    x_before_pad = x
    x = [lstm_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x]
    inputs, _ = lstm_preprocessor.gen_input_and_label(x)
    count_vectors = lstm_preprocessor.gen_count_vectors(x)

    print('== Generate %s targets ==' % mode)
    target_lens = [len(t) - FLAGS.h for t in x]
    target_lens = [t if t > 0 else 0 for t in target_lens]

    targets = [np.array(list(map(lstm_preprocessor.v_map, t[FLAGS.h:])), dtype=np.float64) for t in x]
    for i in range(len(targets)):
        assert targets[i].shape[0] == target_lens[i]

    print('== Start applying model ==')
    results = model.predict_generator(gen_batch(FLAGS.batch_size, inputs, count_vectors),
                                      steps=get_batch_count(inputs, FLAGS.batch_size),
                                      verbose=1)

    print('== Start calculating precision, recall and F-measure ==')

    tp, tn, fp, fn = 0, 0, 0, 0

    target_pos = 0
    for i in range(len(targets)):
        inference = compare(results[target_pos: target_pos + target_lens[i]],
                            targets[i])  # remember that results is an array, while targets is a list of arrays

        target_pos += target_lens[i]

        if inference == 1:
            if y[i] == 1:
                tp += 1
                if collector:
                    collector.add_instance(x_before_pad[i], 'tp')
            else:
                fp += 1
                if collector:
                    collector.add_instance(x_before_pad[i], 'fp')
        else:
            if y[i] == 1:
                fn += 1
                if collector:
                    collector.add_instance(x_before_pad[i], 'fn')
            else:
                tn += 1
                if collector:
                    collector.add_instance(x_before_pad[i], 'tn')

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print('TP: %g' % tp)
    print('TN: %g' % tn)
    print('FP: %g' % fp)
    print('FN: %g' % fn)
    print('Precision: %g' % precision)
    print('Recall: %g' % recall)
    print('F-measure: %g' % f1_score)

    if collector:
        collector.write_collections()

    return f1_score


class ValCallback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.best_f1_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % FLAGS.checkpoint_frequency == 0:
            f1_score = apply_model(self.x_val, self.y_val, self.model, 'validation')
            if f1_score > self.best_f1_score:
                print('Saving model to %s' % checkpoint_name)
                model.save_weights(checkpoint_name)
                self.best_f1_score = f1_score


# datasets = ['BGL', 'HDFS']
datasets = ['HDFS']

if __name__ == '__main__':
    for dataset in datasets:
        print('########### Start LSTM on Dataset ' + dataset + ' ###########')
        config.init('LSTM_' + dataset)
        checkpoint_name = config.path + FLAGS.checkpoint_name

        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = (None, None), (None, None), (None, None)
        collector = None
        if dataset == 'BGL':
            data_instances = config.BGL_data

            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = load_BGL(data_instances, 0.35, 0.6)

        if dataset == 'HDFS':
            data_instances = config.HDFS_data
            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = dataloader.load_HDFS(data_instances,
                                                                                                  train_ratio=0.35,
                                                                                                  is_data_instance=True,
                                                                                                  test_ratio=0.6)
            result_folder = config.path + FLAGS.result_folder
            collector = Collector(result_folder, (1, 1, 1, 1), False, config.HDFS_col_header, 100)

        assert FLAGS.h < FLAGS.plb
        lstm_preprocessor = preprocessing.LstmPreprocessor(x_train, x_test, x_validate)
        sym_count = len(lstm_preprocessor.vectors) - 1
        print('Total symbols: %d' % sym_count)
        print(lstm_preprocessor.syms)

        # don't pad x_train
        # x_train = [lstm_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x_train]

        # throw away same event series in x_train
        x_temp = x_train
        x_train = []
        x_hash = set()
        for i in range(len(y_train)):
            if FLAGS.no_repeat_series == 1:
                for j in range(len(x_temp[i]) - FLAGS.h):
                    s = x_temp[i][j: j + FLAGS.h + 1]
                    hs = hash(str(s))
                    if hs not in x_hash:
                        x_hash.add(hs)
                        x_train.append(s)
            else:
                hs = hash(str(x_temp[i]))
                if hs not in x_hash:
                    x_hash.add(hs)
                    x_train.append(x_temp[i])

        model = lstm_attention_count_vector.LSTMAttention(FLAGS.g, FLAGS.h, FLAGS.L, FLAGS.alpha, FLAGS.batch_size,
                                                          sym_count).model
        # checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name,
        #                                              verbose=1, save_weights_only=True)
        val_callback = ValCallback(x_validate, y_validate)

        if os.path.exists(checkpoint_name):
            print('== Reading model parameters from %s ==' % checkpoint_name)
            model.load_weights(checkpoint_name)

        if FLAGS.epochs > 0:
            print('== Start training ==')

            inputs, labels = lstm_preprocessor.gen_input_and_label(x_train)
            count_vectors = lstm_preprocessor.gen_count_vectors(x_train)
            model.fit_generator(gen_batch(FLAGS.batch_size, inputs, count_vectors, labels, True),
                                steps_per_epoch=get_batch_count(inputs, FLAGS.batch_size),
                                epochs=FLAGS.epochs, verbose=1, callbacks=[val_callback])

        model.load_weights(checkpoint_name)
        apply_model(x_test, y_test, model, 'inference', collector)

import numpy as np
import keras
import tensorflow as tf
import os
import sys

sys.path.append('../')
sys.path.append('/root/')  # for docker
from loglizer import preprocessing
from loglizer.models import lstm_keras
from workflow.BGL_workflow.data_generator import load_BGL
from workflow import dataloader
from scripts import config

import random

flags = tf.app.flags
flags.DEFINE_integer('epochs', 15, 'epochs to train')
flags.DEFINE_integer('batch_size', 15, 'batch size')
flags.DEFINE_integer('g', 5, 'the cutoff in the prediction output to be considered normal')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('L', 2, 'number of layers')
flags.DEFINE_integer('alpha', 64, 'number of memory units')
flags.DEFINE_integer('plb', 12,
                     'padding lower bound, pad to this amount')  # this should be set to prevent length of block < window size, wipe those block with length < window size if this amount is set to 0
flags.DEFINE_string('checkpoint_name', 'lstm_keras.h5', 'training directory')
FLAGS = flags.FLAGS


def get_batch_count(a, batch_size):
    return (len(a) - 1) // batch_size + 1


def gen_batch(batch_size, i, l=None, shuffle=False):
    while True:
        if shuffle:
            randnum = random.randint(0, 10000)
            random.seed(randnum)
            random.shuffle(i)
            if l is not None:
                random.seed(random)
                random.shuffle(l)

        c = get_batch_count(i, batch_size)
        for k in range(0, c * batch_size, batch_size):
            u = len(i) if k + batch_size > len(i) else k + batch_size
            if l is not None:
                yield i[k: u], l[k: u]
            else:
                yield i[k: u]


def compare(output, target):
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
            return 1  # anomaly
    return 0  # normal


def apply_model(x, y, mode='inference'):
    print('== Start %s ==' % mode)
    print('== Generate %s inputs ==' % mode)
    x = [lstm_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x]
    inputs, _ = lstm_preprocessor.gen_input_and_label(x)

    print('== Generate %s targets ==' % mode)
    target_lens = [len(t) - FLAGS.h for t in x]
    target_lens = [t if t > 0 else 0 for t in target_lens]

    targets = [np.array(list(map(lstm_preprocessor.v_map, t[FLAGS.h:])), dtype=np.float64) for t in x]
    for i in range(len(targets)):
        assert targets[i].shape[0] == target_lens[i]

    print('== Start applying model ==')
    results = model.predict_generator(gen_batch(FLAGS.batch_size, inputs),
                                      steps=get_batch_count(inputs, FLAGS.batch_size),
                                      verbose=1)

    print('== Start calculating precision, recall and F-measure ==')

    tot_positives = 0
    tot_anomalies = 0
    precision = 0
    recall = 0

    target_pos = 0
    for i in range(len(targets)):
        if target_lens[i] == 0:
            continue
        inference = compare(results[target_pos: target_pos + target_lens[i]],
                            targets[i])  # remember that results is an array, while targets is a list of arrays
        target_pos += target_lens[i]
        if inference == 1:
            tot_positives += 1
            if y[i] == 1:
                precision += 1
        if y[i] == 1:
            tot_anomalies += 1
            if inference == 1:
                recall += 1

    precision /= tot_positives
    recall /= tot_anomalies
    print('Total positives: %g' % tot_positives)
    print('Total anomalies: %g' % tot_anomalies)
    print('Precision: %g' % precision)
    print('Recall: %g' % recall)
    print('F-measure: %g' % (2 * precision * recall / (precision + recall)))


class ValCallback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        apply_model(self.x_val, self.y_val, 'validation')


# datasets = ['BGL', 'HDFS']
datasets = ['HDFS']


if __name__ == '__main__':
    for dataset in datasets:
        print('########### Start LSTM on Dataset ' + dataset + ' ###########')
        config.init('LSTM_' + dataset)
        checkpoint_name = config.path + FLAGS.checkpoint_name

        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = (None, None), (None, None), (None, None)
        if dataset == 'BGL':
            data_instances = config.BGL_data

            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = load_BGL(data_instances, 0.3, 0.6)

        if dataset == 'HDFS':
            data_instances = config.HDFS_data
            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = dataloader.load_HDFS(data_instances,
                                                                                                  train_ratio=0.3,
                                                                                                  is_data_instance=True,
                                                                                                  test_ratio=0.6)

        lstm_preprocessor = preprocessing.LstmPreprocessor(x_train, x_test, x_validate)
        sym_count = len(lstm_preprocessor.vectors) - 1
        print('Total symbols: %d' % sym_count)
        print(lstm_preprocessor.syms)

        # throw away anomalies in x_train
        x_temp = x_train
        x_train = []
        for i in range(len(y_train)):
            if y_train[i] == 0:
                x_train.append(x_temp[i])
        # pad x_train
        x_train = [lstm_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x_train]

        model = lstm_keras.LSTM(FLAGS.g, FLAGS.h, FLAGS.L, FLAGS.alpha, FLAGS.batch_size, sym_count).model
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name,
                                                     verbose=1, save_weights_only=True)
        val_callback = ValCallback(x_validate, y_validate)

        if os.path.exists(checkpoint_name):
            print('== Reading model parameters from %s ==' % checkpoint_name)
            model.load_weights(checkpoint_name)

        if FLAGS.epochs > 0:
            print('== Start training ==')

            inputs, labels = lstm_preprocessor.gen_input_and_label(x_train)
            avg_loss = 0
            model.fit_generator(gen_batch(FLAGS.batch_size, inputs, labels, True),
                                steps_per_epoch=get_batch_count(inputs, FLAGS.batch_size),
                                epochs=FLAGS.epochs, verbose=1, callbacks=[checkpoint, val_callback])

        apply_model(x_test, y_test, 'inference')

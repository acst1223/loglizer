import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import os
import sys
import pickle

sys.path.append('../')
sys.path.append('/root/')  # for docker
from loglizer import preprocessing
from loglizer.models import lstm_attention_count_vector
from workflow.BGL_workflow.data_generator import load_BGL
from workflow import dataloader
from testbed import config
from collector.collector import Collector

flags = tf.app.flags
flags.DEFINE_integer('epochs', 20, 'epochs to train')
flags.DEFINE_integer('batch_size', 128, 'batch size')
# flags.DEFINE_integer('g', 5, 'the cutoff in the prediction output to be considered normal')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('L', 2, 'number of layers')
flags.DEFINE_integer('alpha', 64, 'number of memory units')
flags.DEFINE_integer('plb', 11,
                     'padding lower bound, pad to this amount')  # this should be set to prevent length of block < window size, wipe those block with length < window size if this amount is set to 0
flags.DEFINE_string('checkpoint_name', 'testbed.h5', 'training directory')
flags.DEFINE_string('result_folder', 'result', 'folder to save results')
flags.DEFINE_float('max_mismatch_rate', 0, 'max rate of mismatch tolerated')
# flags.DEFINE_integer('no_repeat_series', 1, 'whether series will not be repeated: 1: no repeat; 0: repeat')
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


def get_top_counts(output, target):
    assert len(output) == len(target)
    result = []
    for i in range(len(target)):
        t = np.sum(output[i] * target[i])
        result.append(t)
    return result


if __name__ == '__main__':
    assert FLAGS.h < FLAGS.plb

    config.init('testbed')
    checkpoint_name = config.path + FLAGS.checkpoint_name
    top_counts_file_name = config.path + 'top_counts.pkl'

    file = config.testbed_path + 'logstash-2019.07.22_ts-food-service_sorted.csv_structured.csv'
    df = pd.read_csv(file)
    event_sequence = [list(df['EventId'].values)]

    lstm_preprocessor = preprocessing.LstmPreprocessor(event_sequence)
    sym_count = len(lstm_preprocessor.vectors) - 1
    print('Total symbols: %d' % sym_count)
    print(lstm_preprocessor.syms)

    model = lstm_attention_count_vector.LSTMAttention(3, FLAGS.h, FLAGS.L, FLAGS.alpha, FLAGS.batch_size,
                                                      sym_count).model
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=1, save_weights_only=True)

    if os.path.exists(checkpoint_name):
        print('== Reading model parameters from %s ==' % checkpoint_name)
        model.load_weights(checkpoint_name)

    inputs, labels = lstm_preprocessor.gen_input_and_label(event_sequence)
    count_vectors = lstm_preprocessor.gen_count_vectors(event_sequence)

    if FLAGS.epochs > 0:
        print('== Start training ==')
        model.fit_generator(gen_batch(FLAGS.batch_size, inputs, count_vectors, labels, True),
                            steps_per_epoch=get_batch_count(inputs, FLAGS.batch_size),
                            epochs=FLAGS.epochs, verbose=1, callbacks=[checkpoint])

    model.load_weights(checkpoint_name)

    print('== Start predicting ==')
    outputs = model.predict_generator(gen_batch(FLAGS.batch_size, inputs, count_vectors),
                                      steps=get_batch_count(inputs, FLAGS.batch_size))
    top_counts = get_top_counts(outputs, labels)
    with open(top_counts_file_name, 'wb') as f:
        pickle.dump(top_counts, f)

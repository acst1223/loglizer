import numpy as np
import keras
import tensorflow as tf
import os
import sys
import math

sys.path.append('../')
sys.path.append('/root/')  # for docker
from loglizer import preprocessing, dataloader, config
from loglizer.models.vae_lstm2 import VAELSTM
from loglizer.data_generator import load_BGL
from collector.collector import Collector
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

flags = tf.compat.v1.app.flags
flags.DEFINE_integer('epochs', 60, 'epochs to train')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('z_dim', 20, 'dimension of z')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('alpha', 256, 'number of memory units')
flags.DEFINE_integer('plb', 10,
                     'padding lower bound, pad to this amount')
flags.DEFINE_string('checkpoint_name', 'vae_lstm2.h5', 'training directory')
flags.DEFINE_string('result_folder', 'result', 'folder to save results')
flags.DEFINE_float('max_mismatch_rate', 0, 'max rate of mismatch tolerated')
flags.DEFINE_integer('no_repeat_series', 0, 'whether series will not be repeated: 1: no repeat; 0: repeat')
flags.DEFINE_integer('checkpoint_frequency', 3, 'every ? epochs to save checkpoints')
flags.DEFINE_string('dataset', 'HDFS', 'name of the dataset')
FLAGS = flags.FLAGS


def padding_zero(a, batch_size):
    '''
    pad zero to an array
    :param a: numpy array, shape: (batch, h, sym_count)
    :return: shape(batch', h, sym_count), batch' is the smallest number that could by divided by batch_size and >= batch
    '''
    amount = 0 if a.shape[0] % batch_size == 0 else batch_size - a.shape[0] % batch_size
    if amount == 0:
        return a
    return np.concatenate((a, np.zeros(shape=(amount,) + a.shape[1:])))


def gen_batch(batch_size, i, is_train=False):
    while True:
        if is_train:
            np.random.shuffle(i)

        c = len(i) // batch_size
        for k in range(0, c * batch_size, batch_size):
            yield i[k: k + batch_size], i[k: k + batch_size, -1]

        if not is_train:
            yield padding_zero(i[c * batch_size:], batch_size), padding_zero(i[c * batch_size:], batch_size)[:, -1]


def decide_boundary(ll_n, ll_a):
    '''
    When evaluating, we have two lists of ll (log likelihood):
    log likelihood according to normal and anomalous instances.
    What we need to do is to decide a boundary value:
    Instances with ll above this boundary value will be considered normal,
    while the others will be considered anomalous.
    The best boundary should be the one that would yield a best F1-Score.

    Args:
        ll_n: List of log likelihood of normal instances.
        ll_a: List of log likelihood of anomalous instances.

    Returns:
        boundary: A float that would yield a best F1-Score.
        precision according to the boundary
        recall according to the boundary
        F1-Score according to the boundary
    '''
    l_n, l_a = len(ll_n), len(ll_a)
    assert l_a > 0
    ll_n = sorted(ll_n)
    ll_a = sorted(ll_a)

    # Assume that the number of normal instances with ll <= boundary is k',
    # the number of anomalous instances with ll <= boundary is k,
    # and the total number of anomalous instances is l_a,
    # then maximize F1-Score is equivalent to minimizing (l_a + k') / k.
    #
    # In this function, k' == i_n, k == i_a + 1
    i_n = 0
    m = float('inf')
    boundary, precision, recall, f1_score = None, None, None, None
    for i_a in range(l_a):
        while i_n < l_n and ll_n[i_n] <= ll_a[i_a]:
            i_n += 1
        m_tmp = (l_a + i_n) / (i_a + 1)
        if m_tmp < m:
            m = m_tmp
            boundary = ll_a[i_a]
            precision = (i_a + 1) / (i_n + i_a + 1)
            recall = (i_a + 1) / l_a
            f1_score = 2 * precision * recall / (precision + recall)
    return boundary, precision, recall, f1_score


def get_normal_anomaly_ll(ll, y, x_count):
    assert 0 <= len(ll) - np.sum(np.array(x_count)) < FLAGS.batch_size
    normal_ll = []
    anomaly_ll = []
    idx = 0
    for i, c in enumerate(x_count):
        if y[i] == 0:
            normal_ll.append(np.min(ll[idx: idx + c]))
        else:
            anomaly_ll.append(np.min(ll[idx: idx + c]))
        idx += c
    return normal_ll, anomaly_ll


def stat_normal_anomaly_ll(ll, y, x_count):
    normal_ll, anomaly_ll = get_normal_anomaly_ll(ll, y, x_count)
    return decide_boundary(normal_ll, anomaly_ll)


class ValCallback(keras.callbacks.Callback):
    def __init__(self, nll_model, x_val, x_val_count, y_val):
        super().__init__()
        self.nll_model = nll_model
        self.x_val = x_val
        self.x_val_count = x_val_count
        self.y_val = y_val
        self.boundary = None
        self.best_f1_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % FLAGS.checkpoint_frequency == 0:
            ll = -self.nll_model.predict_generator(gen_batch(FLAGS.batch_size, self.x_val),
                                              steps=math.ceil(len(self.x_val) / FLAGS.batch_size))
            boundary, precision, recall, f1_score = stat_normal_anomaly_ll(ll, self.y_val, self.x_val_count)
            print('After epoch %d: Boundary: %g, Precision: %g, Recall: %g, F1-Score: %g' % (
                epoch + 1, boundary, precision, recall, f1_score
            ))
            if f1_score > self.best_f1_score:
                print('Saving model to %s' % checkpoint_name)
                model.save_weights(checkpoint_name)
                self.boundary = boundary
                self.best_f1_score = f1_score


if __name__ == '__main__':
    dataset = FLAGS.dataset
    print('########### Start VAE with LSTM on Dataset ' + dataset + ' ###########')
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

    assert FLAGS.h <= FLAGS.plb
    vae_preprocessor = preprocessing.VAEPreprocessor(x_train, x_test, x_validate)
    sym_count = len(vae_preprocessor.vectors) - 1
    print('Total symbols: %d' % sym_count)
    print(vae_preprocessor.syms)

    # don't pad x_train
    # x_train = [vae_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x_train]

    # throw away anomalies & same event series in x_train
    x_temp = x_train
    x_train = []
    x_hash = set()
    for i in range(len(y_train)):
        # if y_train[i] == 0:   # Unsupervised. Do not throw away anomalies.
        if FLAGS.no_repeat_series == 1:
            for j in range(len(x_temp[i]) - FLAGS.h + 1):
                s = x_temp[i][j: j + FLAGS.h]
                hs = hash(str(s))
                if hs not in x_hash:
                    x_hash.add(hs)
                    x_train.append(s)
        else:
            hs = hash(str(x_temp[i]))
            if hs not in x_hash:
                x_hash.add(hs)
                x_train.append(x_temp[i])

    v = VAELSTM(FLAGS.h, sym_count, FLAGS.batch_size, FLAGS.z_dim, FLAGS.alpha)
    model = v.model
    nll_model = v.nll_model

    x_validate = [vae_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x_validate]
    x_validate_count = vae_preprocessor.gen_count_of_sequence(x_validate)
    x_validate_inputs = vae_preprocessor.gen_inputs(x_validate)
    val_callback = ValCallback(nll_model, x_validate_inputs, x_validate_count, y_validate)

    if os.path.exists(checkpoint_name):
        print('== Reading model parameters from %s ==' % checkpoint_name)
        model.load_weights(checkpoint_name)

    if FLAGS.epochs > 0:
        print('== Start training ==')

        inputs = vae_preprocessor.gen_inputs(x_train)
        model.fit_generator(gen_batch(FLAGS.batch_size, inputs, is_train=True),
                            steps_per_epoch=len(inputs) // FLAGS.batch_size,
                            epochs=FLAGS.epochs, verbose=1, callbacks=[val_callback])

    if val_callback.best_f1_score == 0.0:
        val_callback.on_epoch_end(epoch=0)

    model.load_weights(checkpoint_name)

    x_test = [vae_preprocessor.pad(t, FLAGS.plb) if len(t) < FLAGS.plb else t for t in x_test]
    x_test_inputs = vae_preprocessor.gen_inputs(x_test)
    ll_test = -nll_model.predict_generator(gen_batch(FLAGS.batch_size, x_test_inputs),
                                           steps=math.ceil(len(x_test_inputs) / FLAGS.batch_size))
    x_test_count = vae_preprocessor.gen_count_of_sequence(x_test)
    normal_ll, anomaly_ll = get_normal_anomaly_ll(ll_test, y_test, x_test_count)
    boundary = val_callback.boundary
    tp, fp = 0, 0
    for item in normal_ll:
        if item <= boundary:
            fp += 1
    for item in anomaly_ll:
        if item <= boundary:
            tp += 1
    precision = tp / (tp + fp)
    recall = tp / len(anomaly_ll)
    f1_score = 2 * precision * recall / (precision + recall)
    print('Test: Precision: %g, Recall: %g, F1-Score: %g' % (precision, recall, f1_score))

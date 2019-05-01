import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import sys
sys.path.append('../')
sys.path.append('/root/') # for docker
from loglizer import dataloader, preprocessing
from loglizer.models import LSTM
import random
from scripts import config
from scipy import stats

flags = tf.app.flags
flags.DEFINE_integer('epochs', 10, 'epochs to train')
flags.DEFINE_integer('epoch_base', 0, 'base of epoch')
flags.DEFINE_integer('bat ch_size', 15, 'batch size')
flags.DEFINE_integer('g', 7, 'the cutoff in the prediction output to be considered normal')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('L', 2, 'number of layers')
flags.DEFINE_integer('alpha', 64, 'number of memory units')
flags.DEFINE_integer('plb', 12, 'padding lower bound, pad to this amount') # this should be set to prevent length of block < window size, wipe those block with length < window size if this amount is set to 0
flags.DEFINE_string('train_dir', 'train', 'training directory')
flags.DEFINE_integer('inference_version', -1, 'version for inference') # use latest if == -1
FLAGS = flags.FLAGS


def padding_zero(a):
    '''
    pad zero to an array
    :param a: numpy array, shape: (batch, h, sym_count)
    :return: shape(batch', h, sym_count), batch' is the smallest number that could by divided by batch_size and >= batch
    '''
    amount = 0 if a.shape[0] % FLAGS.batch_size == 0 else FLAGS.batch_size - a.shape[0] % FLAGS.batch_size
    if amount == 0:
        return a
    return np.concatenate((a, np.zeros(shape=(amount,) + a.shape[1:]))), amount


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
    for item in x:
        if len(item) < FLAGS.plb:
            lstm_preprocessor.pad(item, FLAGS.plb)
    inputs, _ = lstm_preprocessor.gen_input_and_label(x)
    inputs = padding_zero(inputs)

    print('== Generate %s targets ==' % mode)
    target_lens = [len(t) - FLAGS.h for t in x]
    target_lens = [t if t > 0 else 0 for t in target_lens]

    targets = [np.array(list(map(lstm_preprocessor.v_map, t[FLAGS.h:])), dtype=np.float64) for t in x]
    for i in range(len(targets)):
        assert targets[i].shape[0] == target_lens[i]

    print('== Start applying model ==')
    results = []
    for k in range(int(np.shape(inputs)[0] / FLAGS.batch_size)):
        results.append(model.inference(sess, inputs[k * FLAGS.batch_size: (k + 1) * FLAGS.batch_size]))

    results = np.concatenate(tuple(results))

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
    config.log('Total positives: %g' % tot_positives)
    config.log('Total anomalies: %g' % tot_anomalies)
    config.log('Precision: %g' % precision)
    config.log('Recall: %g' % recall)
    config.log('F-measure: %g' % (2 * precision * recall / (precision + recall)))


if __name__ == '__main__':
    config.init('LSTM_HDFS')
    data_instances = config.HDFS_data
    train_dir = config.path + FLAGS.train_dir
    (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = dataloader.load_HDFS(data_instances,
                                                                    train_ratio=0.3,
                                                                    is_data_instance=True,
                                                                    test_ratio=0.6)
    lstm_preprocessor = preprocessing.LstmPreprocessor(x_train, x_test, x_validate)
    sym_count = len(lstm_preprocessor.vectors) - 1
    print('Total symbols: %d' % sym_count)

    # throw away anomalies in x_train
    x_temp = x_train
    x_train = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            x_train.append(x_temp[i])

    with tf.Session() as sess:

        model = LSTM.LSTM(FLAGS.g, FLAGS.h, FLAGS.L, FLAGS.alpha, FLAGS.batch_size, sym_count)

        if tf.train.get_checkpoint_state(train_dir):
            print('== Reading model parameters from %s ==' % train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            print('== Generating new parameters ==')
            tf.global_variables_initializer().run()

        if FLAGS.epochs > 0:
            print('== Start training ==')

        for epoch_i in range(FLAGS.epochs):
            epoch = FLAGS.epoch_base + epoch_i

            config.log('== Epoch %d ==' % epoch)

            # shuffle
            randnum = random.randint(0, 10000)
            random.seed(randnum)
            random.shuffle(x_train)
            random.seed(random)
            random.shuffle(y_train)

            inputs, labels = lstm_preprocessor.gen_input_and_label(x_train)
            avg_loss = 0
            for k in range(int(np.shape(inputs)[0] / FLAGS.batch_size)):
                loss, _ = model.train(sess,
                                      inputs[k * FLAGS.batch_size: (k + 1) * FLAGS.batch_size],
                                      labels[k * FLAGS.batch_size: (k + 1) * FLAGS.batch_size])
                avg_loss += loss

            avg_loss /= int(np.shape(x_train)[0] / FLAGS.batch_size)
            print('avg loss: %g' % avg_loss)
            config.log('avg loss: %g' % avg_loss)

            model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=epoch)

            apply_model(x_validate, y_validate, 'validating')

        if FLAGS.inference_version != -1:
            model.saver.restore(sess, train_dir + '/' + 'checkpoint-%08d' % FLAGS.inference_version)

        apply_model(x_test, y_test, 'inference')

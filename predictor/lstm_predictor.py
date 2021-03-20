import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import sys

sys.path.append('../')
sys.path.append('/root/')  # for docker
from workflow.baidu_workflow import preprocessing
from loglizer.models import LSTM
import os

import random

flags = tf.compat.v1.app.flags
flags.DEFINE_integer('epochs', 0, 'epochs to train')
flags.DEFINE_integer('epoch_base', 66, 'base of epoch')
flags.DEFINE_integer('batch_size', 15, 'batch size')
flags.DEFINE_integer('g', 5, 'the cutoff in the prediction output to be considered normal')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('L', 2, 'number of layers')
flags.DEFINE_integer('alpha', 64, 'number of memory units')
flags.DEFINE_integer('template_count', 300, 'number of templates')
flags.DEFINE_integer('inference_version', -1, 'version for inference, use latest if == -1')
flags.DEFINE_string('checkpoint_dir', 'lstm_predictor', 'directory to store checkpoints')
flags.DEFINE_string('preprocessor_process_file', 'preprocessor_process.pkl', 'pickle file to save process of preprocessor')
FLAGS = flags.FLAGS


def compare(output, target):
    t = np.sum(output * target)
    s = list(output)
    s = sorted(s, reverse=True)
    found = False
    for j in range(FLAGS.g):
        if s[j] == t:
            found = True
            break
    if not found:
        return 1  # anomaly
    return 0  # normal


if __name__ == '__main__':
    preprocessor = preprocessing.Preprocessor('../data/baidu/train', '../data/baidu/test', FLAGS.batch_size,
                                              FLAGS.h, FLAGS.template_count)

    with tf.compat.v1.Session() as sess:

        model = LSTM.LSTM(FLAGS.g, FLAGS.h, FLAGS.L, FLAGS.alpha, FLAGS.batch_size, FLAGS.template_count)

        if tf.train.get_checkpoint_state(FLAGS.checkpoint_dir):
            print('== Reading model parameters from %s ==' % FLAGS.checkpoint_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            preprocessor.load_process(FLAGS.preprocessor_process_file)
        else:
            print('== Generating new parameters ==')
            tf.compat.v1.global_variables_initializer().run()

        if FLAGS.epochs > 0:
            print('== Start training ==')

        for epoch_i in range(FLAGS.epochs):
            epoch = FLAGS.epoch_base + epoch_i

            print('== Epoch %d ==' % epoch)

            preprocessor.gen_train_info()
            preprocessor.save_process(FLAGS.preprocessor_process_file)

            avg_loss = 0
            for inputs, labels in preprocessor.get_train_flow().get_batch():
                loss, _ = model.train(sess, inputs, labels)
                avg_loss += loss
            avg_loss /= int(len(preprocessor.inputs) / preprocessor.batch_size)
            print('avg loss: %g' % avg_loss)

            model.saver.save(sess, '%s/checkpoint' % FLAGS.checkpoint_dir, global_step=epoch)

        if FLAGS.inference_version != -1:
            model.saver.restore(sess, FLAGS.checkpoint_dir + '/' + 'checkpoint-%08d' % FLAGS.inference_version)

        for i in preprocessor.test_loop():
            print('== Test Loop %d ==' % i)

            anomaly_times = []

            preprocessor.gen_test_info()

            results = []
            for inputs, labels in preprocessor.get_test_flow().get_batch():
                results.append(model.inference(sess, inputs))
            results = np.concatenate(tuple(results))

            for j, label in enumerate(preprocessor.labels):
                if compare(results[j], label) == 1:
                    anomaly_times.append(preprocessor.times[j])

            with open('result/%s' % os.path.split(preprocessor.test_files[i])[1], 'w') as f:
                for a in anomaly_times:
                    f.write('%d\n' % a)

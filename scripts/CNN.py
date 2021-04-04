import numpy as np
import tensorflow as tf
import sys
import tqdm

sys.path.append('../')
sys.path.append('/root/')  # for docker
from loglizer import preprocessing, dataloader, config
from loglizer.data_generator import load_BGL
from loglizer.models import CNN
import random

flags = tf.compat.v1.app.flags
flags.DEFINE_integer('epochs', 10, 'epochs to train')
flags.DEFINE_integer('epoch_base', 0, 'base of epoch')
flags.DEFINE_integer('log_len', 50, 'default length of log')  # if length < log_len, it will be padded by 0
flags.DEFINE_string('train_dir', 'train', 'training directory')
flags.DEFINE_integer('inference_version', -1, 'version for inference')  # use latest if == -1
flags.DEFINE_string('dataset', 'HDFS', 'name of the dataset')
FLAGS = flags.FLAGS


def apply_model(cnn_preprocessor, x, y, mode='inference'):
    print('== Start %s ==' % mode)
    if mode == 'inference':
        config.log('== Start %s ==' % mode)
    print('== Generate %s inputs ==' % mode)
    inputs = cnn_preprocessor.gen_input(x)
    y = np.array(y, dtype=np.float32)

    print('== Start applying model ==')
    results = []
    for i in range(x.shape[0]):
        results.append(model.inference(sess, inputs[i: i + 1]))
    results = np.array(results, dtype=np.float32)

    print('== Start calculating precision, recall and F-measure ==')

    TP = np.sum(results * y)
    precision = TP / np.sum(results)
    recall = TP / np.sum(y)
    print('Precision: %g' % precision)
    print('Recall: %g' % recall)
    print('F-measure: %g' % (2 * precision * recall / (precision + recall)))
    config.log('Precision: %g' % precision)
    config.log('Recall: %g' % recall)
    config.log('F-measure: %g' % (2 * precision * recall / (precision + recall)))


if __name__ == '__main__':
    dataset = FLAGS.dataset
    print('########### Start CNN on Dataset ' + dataset + ' ###########')
    config.init('CNN_' + dataset)
    train_dir = config.path + FLAGS.train_dir

    if dataset == 'HDFS':
        data_instances = config.HDFS_data
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = dataloader.load_HDFS(data_instances,
                                                                                              train_ratio=0.3,
                                                                                              is_data_instance=True,
                                                                                              test_ratio=0.6,
                                                                                              CNN_option=True)
    elif dataset == 'BGL':
        data_instances = config.BGL_data

        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = load_BGL(data_instances, 0.35, 0.6)

    cnn_preprocessor = preprocessing.CNNPreprocessor(FLAGS.log_len, x_train, x_test, x_validate)
    sym_count = len(cnn_preprocessor.syms) - 1
    print('Total symbols: %d' % sym_count)

    with tf.compat.v1.Session() as sess:

        model = CNN.CNN(sym_count, FLAGS.log_len)

        if tf.train.get_checkpoint_state(train_dir):
            print('== Reading model parameters from %s ==' % train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            print('== Generating new parameters ==')
            tf.compat.v1.global_variables_initializer().run()

        print('== Start training ==')
        for epoch_i in range(FLAGS.epochs):
            epoch = FLAGS.epoch_base + epoch_i

            print('== Epoch %d ==' % epoch)
            config.log('== Epoch %d ==' % epoch)

            # shuffle
            randnum = random.randint(0, 10000)
            random.seed(randnum)
            random.shuffle(x_train)
            random.seed(random)
            random.shuffle(y_train)
            y = cnn_preprocessor.gen_label(y_train)
            np.set_printoptions(threshold=np.inf)

            inputs = cnn_preprocessor.gen_input(x_train)
            avg_loss = 0
            for k in tqdm.trange(np.shape(inputs)[0]):
                loss, _ = model.train(sess,
                                      inputs[k: k + 1],
                                      y[k: k + 1])
                avg_loss += loss

            avg_loss /= np.shape(x_train)[0]
            print('avg loss: %g' % avg_loss)
            config.log('avg loss: %g' % avg_loss)

            model.saver.save(sess, '%s/checkpoint' % train_dir, global_step=epoch)

            apply_model(cnn_preprocessor, x_validate, y_validate, 'validating')

        if FLAGS.inference_version != -1:
            model.saver.restore(sess, train_dir + '/' + 'checkpoint-%08d' % FLAGS.inference_version)

        apply_model(cnn_preprocessor, x_test, y_test, 'inference')

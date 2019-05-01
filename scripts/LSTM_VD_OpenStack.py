import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import sys
sys.path.append('../')
sys.path.append('/root/') # for docker
from loglizer.vd_workflow import vd_dataloader, valid_template
from loglizer.models import LSTM_VD
from scripts import config
from scipy import stats
import math
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


flags = tf.app.flags
flags.DEFINE_integer('epochs', 15, 'epochs to train')
flags.DEFINE_integer('epoch_base', 0, 'base of epoch')
flags.DEFINE_integer('batch_size', 15, 'batch size')
flags.DEFINE_integer('h', 10, 'window size')
flags.DEFINE_integer('L', 2, 'number of layers')
flags.DEFINE_integer('alpha', 64, 'number of memory units')
flags.DEFINE_string('train_dir', 'LSTM_VD_OpenStack', 'training directory')
flags.DEFINE_integer('inference_version', -1, 'version for inference') # use latest if == -1
# now just plot
# TODO: flags.DEFINE_float('conf_level', 0.98, 'confidence level for gaussian distribution of mse')
FLAGS = flags.FLAGS


if __name__ == '__main__':
    config.init('LSTM_VD_OpenStack')
    base_name = config.OpenStack_vd_data
    v_template = valid_template.load_valid_template(config.OpenStack_valid_template)
    train_dir = config.path + FLAGS.train_dir

    dataloaders = {}
    for template in v_template:
        dataloaders[template] = vd_dataloader.VD_Dataloader(base_name, template, FLAGS.h, FLAGS.batch_size,
                                                            train_word='normal2', validate_word='normal1',
                                                            test_word='abnormal')

    with tf.Session() as sess:
        lstms = {}
        for template in v_template:
            lstms[template] = LSTM_VD.LSTM_VD(template, dataloaders[template].argc, FLAGS.h, FLAGS.L, FLAGS.alpha,
                                              FLAGS.batch_size)

        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10, pad_step_number=True)

        if tf.train.get_checkpoint_state(train_dir):
            print('== Reading model parameters from %s ==' % train_dir)
            saver.restore(sess, tf.train.latest_checkpoint(train_dir))
        else:
            print('== Generating new parameters ==')
            tf.global_variables_initializer().run()

        if FLAGS.epochs > 0:
            print('== Start training ==')

        for epoch_i in range(FLAGS.epochs):
            epoch = FLAGS.epoch_base + epoch_i

            config.log('== Epoch %d ==' % epoch)
            tot_loss = 0

            for template in dataloaders.keys():
                dataloader = dataloaders[template]
                dataloader.shuffle()
                avg_loss = 0
                model = lstms[template]
                for input, target in dataloader.gen_train():
                    loss, _ = model.train(sess, input, target)
                    avg_loss += loss
                avg_loss /= int(np.shape(dataloader.train_inputs)[0] / FLAGS.batch_size)
                tot_loss += avg_loss

            config.log('loss: %g' % tot_loss)
            saver.save(sess, '%s/checkpoint' % train_dir, global_step=epoch)

        if FLAGS.inference_version != -1:
            saver.restore(sess, train_dir + '/' + 'checkpoint-%08d' % FLAGS.inference_version)

        print('== Start validating ==')
        distributions = {}
        for template in dataloaders.keys():
            distribution = []
            dataloader = dataloaders[template]
            model = lstms[template]
            for input, target in dataloader.gen_validate():
                distribution.append(model.inference(sess, input, target))
            distribution = dataloader.validate_clip(np.concatenate(tuple(distribution)))
            mean, std = np.mean(distribution), np.std(distribution)
            config.log('%s mean, std: %g, %g' % (template, mean, std))
            distributions[template] = (mean, std)

        print('== Start inference ==')
        formatter = DateFormatter('%H:%M')
        for template in dataloaders.keys():
            dataloader = dataloaders[template]
            model = lstms[template]
            dt = dataloader.test_target_datetime
            output = []
            for input, target in dataloader.gen_test():
                output.append(model.inference(sess, input, target))
            output = dataloader.test_clip(np.concatenate(tuple(output)))
            assert len(dt) == len(output)
            mean, std = distributions[template]
            norm = stats.norm(mean, std)
            output = [-math.log(norm.pdf(t)) if norm.pdf(t) > math.exp(-10) else 10 for t in output]
            ax = plt.subplot(1, 1, 1)
            plt.xlabel('Time')
            plt.ylabel('-log(pdf)')
            plt.plot(dt, output, 'o-')
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            for label in ax.get_xticklabels():
                label.set_rotation(30)
            plt.savefig(config.get_OpenStack_result_png_name(template))
            plt.close()
            config.log('Inference of %s completed' % template)

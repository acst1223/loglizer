import tensorflow as tf
import numpy as np

class LSTM_VD(object):
    '''
    LSTM for variable detection
    '''
    def __init__(self, name, argc, h, L, alpha, batch_size):
        self.name, self.argc, self.h, self.L, self.alpha, self.batch_size = name, argc, h, L, alpha, batch_size

        init = tf.truncated_normal_initializer(stddev=0.01)
        self.scope = 'LSTM_VD_%s' % self.name

        print('== Constructing weights ==')
        with tf.variable_scope(self.scope):
            self.W = tf.get_variable('w', (self.alpha, self.argc), dtype=tf.float64, initializer=init)

        print('== Constructing model ==')
        self.inputs = tf.placeholder(dtype=tf.float64, shape=(self.batch_size, self.h, self.argc))
        self.labels = tf.placeholder(dtype=tf.float64, shape=(self.batch_size, self.argc))
        self.output = self.forward(self.inputs)
        self.batch_err = tf.reduce_mean(tf.square(self.labels - self.output), axis=1)
        self.loss = tf.losses.mean_squared_error(self.labels, self.output)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        ## for inference
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10, pad_step_number=True)

    def forward(self, inputs):
        cells = []
        for i in range(self.L):
            cells.append(tf.nn.rnn_cell.LSTMCell(self.alpha))
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        inputs = tf.unstack(tf.transpose(inputs, (1, 0, 2)))
        initial_state = multi_cell.zero_state(self.batch_size, dtype=tf.float64)
        _, final_hidden_state = tf.nn.static_rnn(multi_cell, inputs, initial_state=initial_state, dtype=tf.float64, scope=self.scope)
        output = tf.matmul(final_hidden_state[self.L - 1][1], self.W)
        return output

    def train(self, sess, inputs, labels):
        return sess.run([self.loss, self.train_op],
                        feed_dict={
                            self.inputs: inputs,
                            self.labels: labels
        })

    # inputs should be padded/cut to batch size
    def inference(self, sess, inputs, labels):
        return sess.run([self.batch_err],
                        feed_dict={
                            self.inputs: inputs,
                            self.labels: labels
        })[0]

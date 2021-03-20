import tensorflow as tf
import numpy as np

class LSTM(object):
    def __init__(self, g, h, L, alpha, batch_size, sym_count, lr=0.001, fc1_size=64):
        self.g, self.h, self.L, self.alpha, self.batch_size, self.sym_count, self.lr = g, h, L, alpha, batch_size, sym_count, lr
        self.fc1_size = fc1_size

        init = tf.compat.v1.truncated_normal_initializer(stddev=0.01)

        print('== Constructing weights ==')
        with tf.compat.v1.variable_scope('LSTM'):
            self.W = tf.compat.v1.get_variable('w', (self.alpha, self.sym_count), dtype=tf.float64, initializer=init)
            # self.W1 = tf.get_variable('w1', (self.alpha, self.fc1_size), dtype=tf.float64, initializer=init)
            # self.W2 = tf.get_variable('w2', (self.fc1_size, self.sym_count), dtype=tf.float64, initializer=init)

        print('== Constructing model ==')
        self.inputs = tf.compat.v1.placeholder(dtype=tf.float64, shape=(self.batch_size, self.h, self.sym_count))
        self.labels = tf.compat.v1.placeholder(dtype=tf.float64, shape=(self.batch_size, self.sym_count))
        output = self.forward(self.inputs)
        self.loss = tf.compat.v1.losses.softmax_cross_entropy(self.labels, output)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
        ## for inference
        self.result = tf.nn.softmax(output)
        self.saver = tf.compat.v1.train.Saver(write_version=tf.compat.v1.train.SaverDef.V2, max_to_keep=10, pad_step_number=True)

    def forward(self, inputs):
        cells = []
        for i in range(self.L):
            cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.alpha))
        multi_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        inputs = tf.unstack(tf.transpose(a=inputs, perm=(1, 0, 2)))
        initial_state = multi_cell.zero_state(self.batch_size, dtype=tf.float64)
        _, final_hidden_state = tf.compat.v1.nn.static_rnn(multi_cell, inputs, initial_state=initial_state, dtype=tf.float64, scope='LSTM')
        output = tf.matmul(final_hidden_state[self.L - 1][1], self.W)
        return output

    def train(self, sess, inputs, labels):
        return sess.run([self.loss, self.train_op],
                        feed_dict={
                            self.inputs: inputs,
                            self.labels: labels
        })

    # inputs should be padded/cut to batch size
    def inference(self, sess, inputs):
        return sess.run([self.result],
                        feed_dict={
                            self.inputs: inputs
        })[0]

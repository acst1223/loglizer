### THIS MODEL IS INCOMPLETE!! ###

import tensorflow as tf
import numpy as np

class CNN(object):
    def __init__(self, sym_count, log_len):
        self.sym_count, self.log_len = sym_count, log_len

        # Config
        init = tf.truncated_normal_initializer(stddev=0.01)
        self.embedding_size = 128

        print('== Constructing weights ==')
        self.embedding_map = tf.get_variable('embedding_map', shape=[self.sym_count, self.embedding_size], initializer=init)
        self.conv1 = tf.get_variable('conv1', (3, 128, 1, 128), dtype=tf.float32, initializer=init)
        self.conv2 = tf.get_variable('conv2', (4, 128, 1, 128), dtype=tf.float32, initializer=init)
        self.conv3 = tf.get_variable('conv3', (5, 128, 1, 128), dtype=tf.float32, initializer=init)
        self.w = tf.get_variable('w', (384, 2), dtype=tf.float32, initializer=init)

        print('== Constructing model ==')
        self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.log_len))
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None, 2))
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.output = self.forward(self.inputs)
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.output)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        ## for inference
        self.result = tf.argmax(tf.nn.softmax(self.output), axis=1)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10, pad_step_number=True)

    def forward(self, inputs):
        embeddings = tf.nn.embedding_lookup(self.embedding_map, inputs)
        embeddings = tf.expand_dims(embeddings, -1)
        c1 = tf.nn.conv2d(embeddings, self.conv1, [1, 1, 1, 1], 'VALID')
        c2 = tf.nn.conv2d(embeddings, self.conv2, [1, 1, 1, 1], 'VALID')
        c3 = tf.nn.conv2d(embeddings, self.conv3, [1, 1, 1, 1], 'VALID')
        p1 = tf.nn.max_pool(tf.nn.leaky_relu(c1, 0.1), [1, 48, 1, 1], [1, 1, 1, 1], 'VALID')
        p2 = tf.nn.max_pool(tf.nn.leaky_relu(c2, 0.1), [1, 47, 1, 1], [1, 1, 1, 1], 'VALID')
        p3 = tf.nn.max_pool(tf.nn.leaky_relu(c3, 0.1), [1, 46, 1, 1], [1, 1, 1, 1], 'VALID')
        cat = tf.reshape(tf.concat([p1, p2, p3], axis=-1), [-1, 384])
        cat = tf.nn.dropout(cat, keep_prob=self.keep_prob)
        fc = tf.matmul(cat, self.w)
        return fc

    def train(self, sess, inputs, labels):
        return sess.run([self.loss, self.train_op],
                        feed_dict={
                            self.inputs: inputs,
                            self.labels: labels,
                            self.keep_prob: 0.5
        })

    def inference(self, sess, inputs):
        result = sess.run([self.result, self.output],
                        feed_dict={
                            self.inputs: inputs,
                            self.keep_prob: 1.0
        })
        # print(result[1])
        return result[0][0]


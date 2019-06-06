import pandas as pd
import os
import numpy as np
import re
import sys
import tensorflow as tf
import pickle
from ast import literal_eval
import glob


class Flow(object):
    def __init__(self, x, batch_size, y=None, pad_end=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.pad_end = pad_end

    def get_batch(self):
        for i in range(0, len(self.x) - (self.batch_size if len(self.x) % self.batch_size != 0 else 0), self.batch_size):
            if self.y is not None:
                yield self.x[i: i + self.batch_size], self.y[i: i + self.batch_size]
            else:
                yield self.x[i: i + self.batch_size]
        if self.pad_end and len(self.x) % self.batch_size != 0:
            if self.y is not None:
                yield self.padding_zero(self.x[len(self.x) - len(self.x) % self.batch_size:]), \
                      self.padding_zero(self.y[len(self.x) - len(self.x) % self.batch_size:])
            else:
                yield self.padding_zero(self.x[len(self.x) - len(self.x) % self.batch_size:])

    def padding_zero(self, a):
        '''
        pad zero to an array
        :param a: numpy array, shape: (batch, h, sym_count)
        :return: shape(batch', h, sym_count), batch' is the smallest number that could by divided by batch_size and >= batch
        '''
        amount = 0 if a.shape[0] % self.batch_size == 0 else self.batch_size - a.shape[0] % self.batch_size
        if amount == 0:
            return a
        return np.concatenate((a, np.zeros(shape=(amount,) + a.shape[1:])))


class Preprocessor(object):
    '''
    gen_xxx_info -> gen_xxx_flow
    '''
    def __init__(self, train_dir, test_dir, batch_size, window_size, template_count):
        self.batch_size = batch_size
        self.window_size = window_size
        self.template_count = template_count
        self.vectors = np.eye(self.template_count, dtype=np.float64)

        self.train_files = [f for f in glob.glob(train_dir + '/*.txt')]
        self.test_files = [f for f in glob.glob(test_dir + '/*.txt')]

        # iterate through all files
        self.train_i = 0
        self.test_i = 0

        self.inputs = None
        self.labels = None
        self.times = None

    def gen_inputs_labels_times(self, file):
        df = pd.read_csv(file)
        data = np.array(df['Template'])
        self.inputs = np.zeros((len(data) - self.window_size, self.window_size, self.template_count))
        self.labels = np.zeros((len(data) - self.window_size, self.template_count))
        for i in range(len(data) - self.window_size):
            self.inputs[i] = self.vectors[data[i: i + self.window_size]]
            self.labels[i] = self.vectors[data[i + self.window_size]]
        self.times = np.array(df['Time'][self.window_size:])

    def gen_train_info(self):
        self.gen_inputs_labels_times(self.train_files[self.train_i])
        self.train_i = (self.train_i + 1) % len(self.train_files)

    def gen_test_info(self):
        self.gen_inputs_labels_times(self.test_files[self.test_i])
        self.test_i = (self.test_i + 1) % len(self.test_files)

    def get_train_flow(self):
        return Flow(self.inputs, self.batch_size, y=self.labels, pad_end=False)

    def get_test_flow(self):
        return Flow(self.inputs, self.batch_size, y=self.labels, pad_end=True)

    def test_loop(self):
        for i, _ in enumerate(self.test_files):
            yield i

    def save_process(self, file):
        with open(file, 'wb') as f:
            pickle.dump((self.train_i, self.test_i), f)

    def load_process(self, file):
        with open(file, 'rb') as f:
            self.train_i, self.test_i = pickle.load(f)

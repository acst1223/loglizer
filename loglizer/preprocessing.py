"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
import sys
from collections import Counter
from scipy.special import expit
from itertools import compress
import tensorflow as tf
import pickle
from ast import literal_eval

class FeatureExtractor(object):

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None
        self.oov = None

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`
            oov: bool, whether to use OOV event
            min_count: int, the minimal occurrence of events (default 0), only valid when oov=True.

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.oov = oov

        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values
        if self.oov:
            oov_vec = np.zeros(X.shape[0])
            if min_count > 1:
                idx = np.sum(X > 0, axis=0) >= min_count
                oov_vec = np.sum(X[:, ~idx] > 0, axis=1)
                X = X[:, idx]
                self.events = np.array(X_df.columns)[idx].tolist()
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X
        
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 
        return X_new

    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters

        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        if self.oov:
            oov_vec = np.sum(X_df[X_df.columns.difference(self.events)].values > 0, axis=1)
            X = np.hstack([X, oov_vec.reshape(X.shape[0], 1)])
        
        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1)) 
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 

        return X_new


class LstmPreprocessor(object):

    def __init__(self, x_train, x_test=None, x_validate=None):
        '''

        :param x_train:
        :param x_test:
        :param x_validate:
        :param stat_path: path of statistic files, pickle formats
        '''

        tot_sym = []
        for item in x_train:
            tot_sym += item
        if not x_test is None:
            for item in x_test:
                tot_sym += item
        if not x_validate is None:
            for item in x_validate:
                tot_sym += item
        self.syms = set(tot_sym)
        self.syms = sorted(list(self.syms)) # Important!!

        self.vectors = {}
        for k, sym in enumerate(self.syms):
            vector = [0 for x in range(len(self.syms))]
            vector[k] = 1
            self.vectors[sym] = vector
        self.vectors['_PAD'] = [0 for x in range(len(self.syms))]

    def pad(self, target, l):
        return ['_PAD'] * (l - len(target)) + target

    def ts_pad(self, target, l):
        return [0] * (l - len(target)) + target

    def v_map(self, sym):
        return self.vectors[sym]

    def gen_input_and_label(self, x):
        inputs, labels = [], []
        FLAGS = tf.flags.FLAGS
        for i in range(len(x)):
            for j in range(len(x[i]) - FLAGS.h):
                inputs.append(list(map(self.v_map, x[i][j: j + FLAGS.h])))
                labels.append(self.v_map(x[i][j + FLAGS.h]))
        inputs = np.array(inputs, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return inputs, labels

    def gen_count_vectors(self, x):
        count_vectors = []
        sym_dict = {sym: i for i, sym in enumerate(self.syms)}
        FLAGS = tf.flags.FLAGS
        for i in range(len(x)):
            for j in range(len(x[i]) - FLAGS.h):
                series = x[i][j: j + FLAGS.h]
                v = [0 for _ in self.syms]
                for sym in series:
                    if sym != '_PAD':
                        v[sym_dict[sym]] += 1
                count_vectors.append(v)
        count_vectors = np.array(count_vectors, dtype=np.float32)
        return count_vectors


class CNNPreprocessor(object):

    def __init__(self, l, x_train, x_test=None, x_validate=None):
        self.l = l
        tot_sym = []
        for item in x_train:
            tot_sym += item
        if not x_test is None:
            for item in x_test:
                tot_sym += item
        if not x_validate is None:
            for item in x_validate:
                tot_sym += item
        tot_sym.append('_PAD')
        self.syms = set(tot_sym)

    def pad(self, target):
        return target + ['_PAD'] * (self.l - len(target))

    def v_map(self, sym):
        return 0 if sym == '_PAD' else int(sym[1:])

    def gen_input(self, x):
        x = [self.pad(t)[: 50] for t in x]
        x = [list(map(self.v_map, t)) for t in x]
        return np.array(x, dtype=np.int32)

    def gen_label(self, y):
        y = np.array(y, dtype=np.int32)
        y = (np.arange(2) == y[:, None]).astype(np.int32) # one-hot encoding
        return y

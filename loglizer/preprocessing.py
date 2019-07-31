import pandas as pd
import random
import numpy as np
from collections import Counter
from scipy.special import expit
import tensorflow as tf
import math


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

        self.hash_dict = dict()
        self.hash_predict_dict = dict()

    def pad(self, target, l):
        return ['_PAD'] * (l - len(target)) + target

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

    def gen_input_and_label_same_length(self, x):
        inputs, labels = [], []
        FLAGS = tf.flags.FLAGS
        for i in range(len(x)):
            inputs.append(list(map(self.v_map, x[i][: FLAGS.h])))
            labels.append(self.v_map(x[i][FLAGS.h]))
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

    def gen_count_vectors_same_length(self, x):
        count_vectors = []
        sym_dict = {sym: i for i, sym in enumerate(self.syms)}
        FLAGS = tf.flags.FLAGS
        for i in range(len(x)):
            series = x[i][: FLAGS.h]
            v = [0 for _ in self.syms]
            for sym in series:
                if sym != '_PAD':
                    v[sym_dict[sym]] += 1
            count_vectors.append(v)
        count_vectors = np.array(count_vectors, dtype=np.float32)
        return count_vectors

    @staticmethod
    def process_train_inputs(x, y, h, throw_away_anomalies, no_repeat_series):
        x_temp = x
        x = []
        x_hash = set()
        for i in range(len(x_temp)):
            if y is None or not throw_away_anomalies or y[i] == 0:
                if no_repeat_series == 1:
                    for j in range(len(x_temp[i]) - h):
                        s = x_temp[i][j: j + h + 1]
                        hs = hash(str(s))
                        if hs not in x_hash:
                            x_hash.add(hs)
                            x.append(s)
                else:
                    hs = hash(str(x_temp[i]))
                    if hs not in x_hash:
                        x_hash.add(hs)
                        x.append(x_temp[i])
        return x

    @staticmethod
    def transform_to_same_length(x, h):
        x_temp = x
        x = []
        for xi in x_temp:
            for j in range(len(xi) - h):
                x.append(xi[j: j + h + 1])
        return x

    @staticmethod
    def get_batch_count(a, batch_size):
        return math.ceil(len(a) / batch_size)

    def gen_batch(self, batch_size, x, with_label, shuffle):
        while True:
            if shuffle:
                random.shuffle(x)
            c = self.get_batch_count(x, batch_size)
            for k in range(0, c * batch_size, batch_size):
                u = len(x) if k + batch_size > len(x) else k + batch_size
                if with_label:
                    yield self.gen_input_and_label_same_length(x[k: u])
                else:
                    yield self.gen_input_and_label_same_length(x[k: u])[0]

    def gen_batch_fast(self, batch_size, x, with_label, shuffle):
        '''
        Faster than gen_batch, but will take up more memory space.
        '''
        inputs, labels = self.gen_input_and_label_same_length(x)

        while True:
            if shuffle:
                randnum = np.random.randint(0, 10000)
                np.random.seed(randnum)
                np.random.shuffle(inputs)
                np.random.seed(randnum)
                np.random.shuffle(labels)

            c = self.get_batch_count(inputs, batch_size)
            for k in range(0, c * batch_size, batch_size):
                u = len(inputs) if k + batch_size > len(inputs) else k + batch_size
                if with_label:
                    yield inputs[k: u], labels[k: u]
                else:
                    yield inputs[k: u]

    def gen_hash_dict(self, x):
        '''
        :param x: Series of templates of same length.
        :return: A list of same length as x, each element of the list is the hash value of the corresponding element
            in x.
        '''
        self.hash_dict = dict()
        result = []
        for xi in x:
            hs = hash(str(xi))
            if hs not in self.hash_dict:
                self.hash_dict[hs] = xi
            result.append(hs)
        return result

    def gen_hash_dict_series(self):
        '''
        :return: A list of all values in self.hash_dict.
        '''
        return [self.hash_dict[k] for k in self.hash_dict]

    def gen_hash_predict_dict(self, x, predictions):
        '''
        x and predictions should be corresponding to each other.
        '''
        self.hash_predict_dict = dict()
        for i, xi in enumerate(x):
            hs = hash(str(xi))
            self.hash_predict_dict[hs] = predictions[i]

    def get_hash_predict_result(self, hash_list):
        return np.vstack([self.hash_predict_dict[i] for i in hash_list])


class VAEPreprocessor(object):

    def __init__(self, x_train, x_test=None, x_validate=None):
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

    def v_map(self, sym):
        return self.vectors[sym]

    def gen_inputs(self, x):
        inputs = []
        FLAGS = tf.flags.FLAGS
        for i in range(len(x)):
            for j in range(len(x[i]) - FLAGS.h + 1):
                inputs.append(list(map(self.v_map, x[i][j: j + FLAGS.h])))
        inputs = np.array(inputs, dtype=np.float32)
        return inputs

    def gen_count_of_sequence(self, x):
        '''
        Note that each event sequence is not of the same length. When raw data become arrays
        or tensors, we cannot tell how to split these arrays/tensors back into event sequences
        without information about length of sequence.
        However, note that when window size(h) is not 1 (under most situations it is much larger
        than 1), the length of event sequence is not equal to length of the array generated from
        that sequence. Therefore get count of sequence returns lengths of the array generated
        from each event sequence (which == length of raw event sequence after padding - h + 1).

        Args:
            x: A list of event sequences after padding.

        Returns:
            A list, each element of the list is length of the array generated from
                each event sequence.
        '''
        return [len(t) - tf.flags.FLAGS.h + 1 for t in x]


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

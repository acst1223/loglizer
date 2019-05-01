import pandas as pd
import pickle
import numpy as np
import random
import datetime
import math


def _padding_zero(a, batch_size):
    amount = 0 if a.shape[0] % batch_size == 0 else batch_size - a.shape[0] % batch_size
    if amount == 0:
        return a
    return np.concatenate((a, np.zeros(shape=(amount,) + a.shape[1:])))


def _df2inputs(df, h, argc):
    '''
    :param df: a DataFrame object
    :param h: window size
    :param argc: argument count
    :return: a 3d matrix (batch x h x argc)
    '''
    l = len(df)
    batch = l - h
    result = np.zeros((batch, h, argc))
    for i in range(batch):
        for j in range(h):
            for k in range(argc):
                result[i, j, k] = df['DeltaTime'][i + j] if k == 0 else df['Arg%d' % (k - 1)][i + j]
    return result


def _df2targets(df, h, argc):
    '''
    :param df: a DataFrame object
    :param h: window size
    :param argc: argument count
    :return: a 2d matrix (batch x argc)
    '''
    l = len(df)
    batch = l - h
    result = np.zeros((batch, argc))
    for i in range(batch):
        for k in range(argc):
            result[i, k] = df['DeltaTime'][i + h] if k == 0 else df['Arg%d' % (k - 1)][i + h]
    return result


class VD_Dataloader(object):
    def __init__(self, base_name, template, h, batch_size,
                 train_word='train', validate_word='validate', test_word='test'):
        self.h = h
        self.template = template
        self.batch_size = batch_size

        train_name = base_name + '_%s_' % train_word + template + '.csv'
        test_name = base_name + '_%s_' % test_word + template + '.csv'
        validate_name = base_name + '_%s_' % validate_word + template + '.csv'
        train_stat = base_name + '_%s_' % train_word + template + '.pkl'

        train_df = pd.read_csv(train_name)
        validate_df = pd.read_csv(validate_name)
        test_df = pd.read_csv(test_name)

        with open(train_stat, 'rb') as f:
            stat = pickle.load(f)
        self.argc = len(stat)

        for df in [train_df, validate_df, test_df]:
            df['DeltaTime'] = (df['DeltaTime'] - stat[0][0]) / (stat[0][1] + 1e-8)
            for i, (mean, std) in enumerate(stat[1:]):
                df['Arg%d' % i] = (df['Arg%d' % i] - mean) / (std + 1e-8)

        self.train_inputs = _df2inputs(train_df, h, self.argc)
        self.validate_inputs = _df2inputs(validate_df, h, self.argc)
        self.validate_cnt = self.validate_inputs.shape[0]
        self.validate_inputs = _padding_zero(self.validate_inputs, self.batch_size)
        self.test_inputs = _df2inputs(test_df, h, self.argc)
        self.test_cnt = self.test_inputs.shape[0]
        self.test_inputs = _padding_zero(self.test_inputs, self.batch_size)

        self.train_targets = _df2targets(train_df, h, self.argc)
        self.validate_targets = _padding_zero(_df2targets(validate_df, h, self.argc), self.batch_size)
        self.test_targets = _padding_zero(_df2targets(test_df, h, self.argc), self.batch_size)

        self.test_target_datetime = (test_df['Date'] + ' ' + test_df['Time'])[h:]
        self.test_target_datetime = [datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in self.test_target_datetime]

        print('== Dataloader init completed: %s ==' % template)

    def shuffle(self):
        randnum = random.randint(0, 10000)
        np.random.seed(randnum)
        np.random.shuffle(self.train_inputs)
        np.random.seed(randnum)
        np.random.shuffle(self.train_targets)

    def gen_train(self):
        for i in range(int(np.shape(self.train_inputs)[0] / self.batch_size)):
            yield self.train_inputs[i * self.batch_size: (i + 1) * self.batch_size], self.train_targets[i * self.batch_size: (i + 1) * self.batch_size]

    def gen_validate(self):
        for i in range(int(np.shape(self.validate_inputs)[0] / self.batch_size)):
            yield self.validate_inputs[i * self.batch_size: (i + 1) * self.batch_size], self.validate_targets[i * self.batch_size: (i + 1) * self.batch_size]

    def gen_test(self):
        for i in range(int(np.shape(self.test_inputs)[0] / self.batch_size)):
            yield self.test_inputs[i * self.batch_size: (i + 1) * self.batch_size], self.test_targets[i * self.batch_size: (i + 1) * self.batch_size]

    def validate_clip(self, validate_inputs):
        return validate_inputs[: self.validate_cnt]

    def test_clip(self, test_inputs):
        return test_inputs[: self.test_cnt]

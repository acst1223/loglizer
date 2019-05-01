import numpy as np
from sklearn.utils import shuffle
from ast import literal_eval
import pandas as pd


def _split_data(x_data, y_data, train_ratio, test_ratio):
    pos_idx = y_data > 0
    x_pos = x_data[pos_idx]
    y_pos = y_data[pos_idx]
    x_neg = x_data[~pos_idx]
    y_neg = y_data[~pos_idx]
    train_pos = int(train_ratio * x_pos.shape[0])
    train_neg = int(train_ratio * x_neg.shape[0])
    test_pos = x_pos.shape[0] - int(test_ratio * x_pos.shape[0])
    test_neg = x_neg.shape[0] - int(test_ratio * x_neg.shape[0])
    x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
    y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
    x_test = np.hstack([x_pos[test_pos:], x_neg[test_neg:]])
    y_test = np.hstack([y_pos[test_pos:], y_neg[test_neg:]])
    x_validate = np.hstack([x_pos[train_pos:test_pos], x_neg[train_neg:test_neg]])
    y_validate = np.hstack([y_pos[train_pos:test_pos], y_neg[train_neg:test_neg]])
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def generate_data(event_sequence_file, train_ratio, test_ratio):
    data_df = pd.read_csv(event_sequence_file)
    data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
    return _split_data(data_df['EventSequence'].values, data_df['Label'].values, train_ratio, test_ratio)

"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
import sys
from sklearn.utils import shuffle
from collections import OrderedDict
from ast import literal_eval
import datetime


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform', test_ratio=None, CNN_option=False):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        if CNN_option:
            train_neg = train_pos # let size of training negatives equal to that of training positives
        else:
            train_neg = int(train_ratio * x_neg.shape[0])
        if test_ratio:
            test_pos = x_pos.shape[0] - int(test_ratio * x_pos.shape[0])
            if CNN_option:
                test_neg = int(train_neg * 1.33)
            else:
                test_neg = x_neg.shape[0] - int(test_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        if test_ratio:
            x_test = np.hstack([x_pos[test_pos:], x_neg[test_neg:]])
            y_test = np.hstack([y_pos[test_pos:], y_neg[test_neg:]])
            x_validate = np.hstack([x_pos[train_pos:test_pos], x_neg[train_neg:test_neg]])
            y_validate = np.hstack([y_pos[train_pos:test_pos], y_neg[train_neg:test_neg]])
        else:
            x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
            y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        if test_ratio:
            num_test = int(test_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        if test_ratio:
            x_test = x_data[-num_test:]
            x_validate = x_data[num_train: -num_test]
        else:
            x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            if test_ratio:
                y_test = y_data[-num_test:]
                y_validate = y_data[num_train: -num_test]
            else:
                y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    if test_ratio:
        return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)
    else:
        return (x_train, y_train), (x_test, y_test), (None, None)


def _split_idxs(y_data, train_ratio, test_ratio=None, CNN_option=False):
    if not test_ratio:
        test_ratio = 1 - train_ratio
    x_data = np.array(list(range(y_data.shape[0])))
    pos_idx = y_data > 0
    x_pos = x_data[pos_idx]
    x_neg = x_data[~pos_idx]
    train_pos = int(train_ratio * x_pos.shape[0])
    if CNN_option:
        train_neg = train_pos  # let size of training negatives equal to that of training positives
    else:
        train_neg = int(train_ratio * x_neg.shape[0])
    test_pos = x_pos.shape[0] - int(test_ratio * x_pos.shape[0])
    if CNN_option:
        test_neg = int(train_neg * 1.33)
    else:
        test_neg = x_neg.shape[0] - int(test_ratio * x_neg.shape[0])
    x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
    x_test = np.hstack([x_pos[test_pos:], x_neg[test_neg:]])
    x_validate = np.hstack([x_pos[train_pos:test_pos], x_neg[train_neg:test_neg]])
    return x_train, x_test, x_validate # remember here x: idxs


def _split_col_header(st):
    cnt = 0 # col header count
    headers = []
    p = re.compile(r'^\d+\.')
    while True:
        pos = st.find('%d.' % (cnt + 2))
        if pos == -1:
            if p.match(st):
                cnt += 1
                p.sub('', st)
                headers.append(st)
            break
        cnt += 1
        headers.append(st[: pos].strip())
        st = st[pos:]

    # wipe numberings
    for i in range(len(headers)):
        while headers[i][0] != '.':
            headers[i] = headers[i][1:]
        headers[i] = headers[i][1:]

    return (headers)

def load_HDFS(log_file, col_header_file=None, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False, is_data_instance=False, test_ratio=None, CNN_option=False):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        is_data_instance: whether the file is data instance, if true, data in data instance will be returned
        test_ratio: ratio of test set(if it is none then there is no validation set)
        CNN_option: if it is True, then split process will be different

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
        (x_validate, y_validate): the validation data
    """

    if CNN_option:
        split_type = 'uniform'

    if is_data_instance:
        assert log_file.endswith('csv')  # data instance file must end with csv
        data_df = pd.read_csv(log_file)
        data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
        # Split train and test data
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(x_data, y_data, train_ratio, split_type,
                                                                                     test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."  # window=session: grouped by EventId
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            if idx % 100000 == 0:
                print("%d rows processed" % idx)
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if not label_file:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _), (x_validate, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type,
                                                                     test_ratio=test_ratio, CNN_option=CNN_option)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_data, None), (x_test, None), (x_validate, None)

    # Warning: Under such circumstances only data instance files will be generated,
    # then the program will quit
    elif log_file.endswith('.log') or log_file.endswith('.txt'):
        if not col_header_file:
            raise FileNotFoundError("Col header file not found!")

        idx = 0
        output_file = log_file[: -3] + 'csv'

        fw = open(output_file, 'w')
        fw.write('LineId,Date,Time,Pid,Level,Component,Content,EventId,EventTemplate,Arg0,Arg1,Arg2,Arg3\n')
        fw.close()

        f = open(col_header_file, 'r')
        text = f.readline()[:-1]
        headers = _split_col_header(text)
        patterns = [re.compile(t) for t in headers]  # patterns according to headers
        f.close()

        # generate number parameter matrix
        npm = np.zeros((29, 4))
        npm[5, 3] = 1
        npm[7, 0] = 1
        npm[8, 1] = 1
        npm[9, 1] = 1
        npm[10, 0] = 1
        npm[14, 1] = 1
        npm[14, 2] = 1
        npm[14, 3] = 1
        npm[25, 2] = 1
        npm[26, 2] = 1
        npm[27, 2] = 1

        f = open(log_file, 'r')
        st = '' # buffer

        while True:
            l = f.readline()[:-1].split(' ')
            if not l or len(l) < 6:
                break
            idx += 1
            st += '%d,' % idx  # LineId
            st += '%s,' % l[0]  # Date
            st += '%s,' % l[1]  # Time
            st += '%s,' % l[2]  # Pid
            st += '%s,' % l[3]  # Level
            st += '%s,' % l[4][: -1]  # Component
            content = ' '.join(l[5:])
            posp = content.find(',')
            if posp != -1:
                content = content[: posp]
            for i in range(len(headers)):
                m = patterns[i].match(content)
                if m:
                    st += '%s,' % content  # Content
                    st += 'E%d,' % i  # EventId
                    st += '%s' % headers[i]  # EventTemplate
                    args = list(m.groups())
                    for j in range(len(args)):
                        if npm[i, j] == 1:
                            st += ','
                            st += args[j]
                    break
            st += '\n'
            if idx % 10000 == 0:
                fw = open(output_file, 'a')
                fw.write(st)
                fw.close()
                st = ''
                print('%d logs converted' % idx)

        f.close()

        if idx % 10000 != 0:
            fw = open(output_file, 'a')
            fw.write(st)
            fw.close()
            st = ''
            print('%d logs converted' % idx)

        return

    else:
        raise NotImplementedError('load_HDFS() only support csv, npz, log and txt files!')

    if test_ratio:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_validate = x_validate.shape[0]
        num_total = num_train + num_test + num_validate
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_validate_pos = sum(y_validate)
        num_pos = num_train_pos + num_test_pos + num_validate_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))
        print('Validate: {} instances, {} anomaly, {} normal\n' \
              .format(num_validate, num_validate_pos, num_validate - num_validate_pos))

    else:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def load_OpenStack(log_file, col_header_file=None, label_file=None, window='session', train_ratio=0.5, split_type='sequential',
    save_csv=False, is_data_instance=False, test_ratio=None, CNN_option=False):
    """ Load OpenStack structured log into train, test and validate data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        is_data_instance: whether the file is data instance, if true, data in data instance will be returned
        test_ratio: ratio of test set(if it is none then there is no validation set)
        CNN_option: if it is True, then split process will be different

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
        (x_validate, y_validate): the validation data
    """

    if CNN_option:
        split_type = 'uniform'

    if is_data_instance:
        assert log_file.endswith('csv')  # data instance file must end with csv
        data_df = pd.read_csv(log_file)
        data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
        # Split train and test data
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(x_data, y_data, train_ratio, split_type,
                                                                                     test_ratio=test_ratio, CNN_option=CNN_option)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for OpenStack dataset."  # window=session: grouped by EventId
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            if idx % 100000 == 0:
                print("%d rows processed" % idx)
            blkId_list = re.findall(r'.*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}).*', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            data_df['Label'] = 0
            fl = open(label_file, 'r')
            while True:
                st = fl.readline()
                if not st:
                    break
                st = st[: -1]
                data_df.ix[data_df['BlockId'] == st, 'Label'] = 1
            fl.close()

            # Split train and test data
            (x_train, y_train), (x_test, y_test), (x_validate, y_validate) = _split_data(data_df['EventSequence'].values,
                data_df['Label'].values, train_ratio, split_type, test_ratio=test_ratio, CNN_option=CNN_option)

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if not label_file:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _), (x_validate, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type,
                                                                     test_ratio=test_ratio, CNN_option=CNN_option)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_data, None), (x_test, None), (x_validate, None)

    # Warning: Under such circumstances only data instance files will be generated,
    # then the program will quit
    elif log_file.endswith('.log') or log_file.endswith('.txt'):
        if not col_header_file:
            raise FileNotFoundError("Col header file not found!")

        idx = 0
        output_file = log_file[: -3] + 'csv'

        fw = open(output_file, 'w')
        fw.write('LineId,Logrecord,Date,Time,Pid,Level,Component,ADDR,Content,EventId,EventTemplate\n')
        fw.close()

        col_df = pd.read_csv(col_header_file)
        raw_headers = col_df['EventTemplate'].tolist()
        headers = [t.replace(r'[', r'\[') for t in raw_headers]
        headers = [t.replace(r']', r'\]') for t in headers]
        headers = [t.replace(r'.', r'\.') for t in headers]
        headers = [t.replace(r'(', r'\(') for t in headers]
        headers = [t.replace(r')', r'\)') for t in headers]
        headers = [t.replace(r'<*>', r'(.*)') for t in headers]
        patterns = [re.compile(t) for t in headers]  # patterns according to headers

        f = open(log_file, 'r')
        st = '' # buffer

        while True:
            st2 = '' # buffer in loop
            l = f.readline()
            if not l:
                break
            l = l[: -1].split(' ')
            if len(l) < 8:
                print("Warning: Invalid format before index %d" % idx)
                continue
            idx += 1
            st2 += '%d,' % idx  # LineId
            st2 += '%s,' % l[0]  # Logrecord
            st2 += '%s,' % l[1]  # Date
            st2 += '%s,' % l[2]  # Time
            st2 += '%s,' % l[3]  # Pid
            st2 += '%s,' % l[4]  # Level
            st2 += '%s,' % l[5]  # Component
            left = ' '.join(l[6:])
            p = left.find(']')
            if p == -1:
                idx -= 1
                print("Warning: Invalid ADDR before index %d" % idx)
                continue
            st2 += '%s,' % left[1: p] # ADDR
            content = left[p + 2:]
            hit = False
            for i in range(len(headers)):
                m = patterns[i].match(content)
                if m:
                    if content.find(',') == -1:
                        st2 += '%s,' % content  # Content
                    else:
                        st2 += '"%s",' % content  # Content
                    st2 += 'E%d,' % i  # EventId
                    if content.find(',') == -1:
                        st2 += '%s' % raw_headers[i]  # EventTemplate
                    else:
                        st2 += '"%s"' % raw_headers[i]  # EventTemplate
                    hit = True
                    break
            if not hit:
                idx -= 1
                continue
            st2 += '\n'
            st += st2
            if idx % 10000 == 0:
                fw = open(output_file, 'a')
                fw.write(st)
                fw.close()
                st = ''
                print('%d logs converted' % idx)

        f.close()

        if idx % 10000 != 0:
            fw = open(output_file, 'a')
            fw.write(st)
            fw.close()
            st = ''
            print('%d logs converted' % idx)

        return

    else:
        raise NotImplementedError('load_OpenStack() only support csv, npz, log and txt files!')

    if test_ratio:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_validate = x_validate.shape[0]
        num_total = num_train + num_test + num_validate
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_validate_pos = sum(y_validate)
        num_pos = num_train_pos + num_test_pos + num_validate_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))
        print('Validate: {} instances, {} anomaly, {} normal\n' \
              .format(num_validate, num_validate_pos, num_validate - num_validate_pos))

    else:
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        num_total = num_train + num_test
        num_train_pos = sum(y_train)
        num_test_pos = sum(y_test)
        num_pos = num_train_pos + num_test_pos
        print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
        print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
        print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def load_BGL(log_file, label_file=None, window='sliding', time_interval=60, stepping_size=60, 
             train_ratio=0.8):
    """  TODO

    """


def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = para['save_path']+'sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n'%inst_number)
        np.savetxt(sliding_file_path,start_end_index_list,delimiter=',',fmt='%d')
    else:
        print('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = [row[0] for row in event_mapping_data]
    event_num = len(list(set(event_mapping_data)))
    print('There are %d log events'%event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    event_count_matrix = np.zeros((inst_number,event_num))
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            event_index = event_mapping_data[k]
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels


def load_HDFS_instances(log_file, train_ratio, test_ratio=None, CNN_option=False):
    data_df = pd.read_csv(log_file)
    data_df['EventSequence'] = data_df['EventSequence'].map(literal_eval)
    data_df['TimeSequence'] = data_df['TimeSequence'].map(literal_eval)
    data_df['PidSequence'] = data_df['PidSequence'].map(literal_eval)
    # Split train and test data
    return data_df, _split_idxs(data_df['Label'].values, train_ratio=train_ratio, test_ratio=test_ratio, CNN_option=CNN_option)

# STEP 1

import sys
sys.path.append('..')
from loglizer import dataloader
import numpy as np
import re
import pandas as pd


def _load_openstack(log_file, col_header_file):
    idx = 0
    output_file = log_file[: -3] + 'csv'

    fw = open(output_file, 'w')
    fw.write('LineId,Date,Time,Instance,EventId,Arg0,Arg1,Arg2,Arg3\n')
    fw.close()

    col_df = pd.read_csv(col_header_file)
    raw_headers = col_df['EventTemplate'].tolist()[: 23]
    headers = [t.replace(r'[', r'\[') for t in raw_headers]
    headers = [t.replace(r']', r'\]') for t in headers]
    headers = [t.replace(r'.', r'\.') for t in headers]
    headers = [t.replace(r'(', r'\(') for t in headers]
    headers = [t.replace(r')', r'\)') for t in headers]
    headers = [t.replace(r'<*>', r'(.*)') for t in headers]
    patterns = [re.compile(t) for t in headers]  # patterns according to headers

    instance_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')

    # generate number parameter matrix
    npm = np.zeros((23, 5))
    npm[0, 1] = 1
    npm[0, 2] = 1
    npm[0, 3] = 1
    npm[9, 1] = 1
    npm[9, 2] = 1
    npm[9, 3] = 1
    npm[9, 4] = 1
    npm[11, 1] = 1
    npm[11, 2] = 1
    npm[12, 1] = 1
    npm[12, 2] = 1
    npm[13, 1] = 1
    npm[13, 2] = 1
    npm[14, 1] = 1
    npm[14, 2] = 1
    npm[15, 1] = 1
    npm[15, 2] = 1
    npm[15, 3] = 1
    npm[16, 1] = 1
    npm[16, 2] = 1
    npm[16, 3] = 1
    npm[17, 1] = 1
    npm[17, 2] = 1
    npm[17, 3] = 1
    '''
    npm = np.zeros((43, 14))
    npm[0, 1] = 1
    npm[0, 2] = 1
    npm[0, 3] = 1
    npm[9, 1] = 1
    npm[9, 2] = 1
    npm[9, 3] = 1
    npm[9, 4] = 1
    npm[11, 1] = 1
    npm[11, 2] = 1
    npm[12, 1] = 1
    npm[12, 2] = 1
    npm[13, 1] = 1
    npm[13, 2] = 1
    npm[14, 1] = 1
    npm[14, 2] = 1
    npm[15, 1] = 1
    npm[15, 2] = 1
    npm[15, 3] = 1
    npm[16, 1] = 1
    npm[16, 2] = 1
    npm[16, 3] = 1
    npm[17, 1] = 1
    npm[17, 2] = 1
    npm[17, 3] = 1
    npm[23, 3] = 1
    npm[23, 4] = 1
    npm[23, 5] = 1
    npm[24, 3] = 1
    npm[24, 4] = 1
    npm[24, 5] = 1
    npm[25, 3] = 1
    npm[25, 4] = 1
    npm[25, 5] = 1
    npm[34, 2] = 1
    npm[34, 3] = 1
    npm[37, 13] = 1
    npm[40, 0] = 1
    npm[40, 1] = 1
    npm[42, 0] = 1
    npm[42, 1] = 1
    '''

    f = open(log_file, 'r')
    st = ''  # buffer

    while True:
        st2 = ''  # buffer in loop
        try:
            l = f.readline()
        except Exception:
            print('Exception met after index %d' % idx)
            continue
        if not l:
            break
        l = l[: -1].split(' ')
        if len(l) < 8:
            print('Line format illegal after index %d' % idx)
            continue
        idx += 1
        st2 += '%d,' % idx  # LineId
        st2 += '%s,' % l[1]  # Date
        st2 += '%s,' % l[2]  # Time

        # get content without address
        content = ' '.join(l[6:])
        p = content.find('-]')
        if p == -1:
            print('ADDR illegal before index %d' % idx)
            idx -= 1
            continue
        content = content[p + 3:]

        instance_legal = True
        match = False
        for i in range(len(headers)):
            m = patterns[i].match(content)
            if m:
                args = list(m.groups())
                st2 += '%s,' % args[0]  # Instance
                instance_legal = instance_pattern.match(args[0])
                st2 += 'E%d' % (i + 1)  # EventId

                for j in range(1, len(args)):
                    if npm[i, j] == 1:
                        st2 += ','
                        st2 += args[j]
                match = True
                break
        if not match:
            idx -= 1
            continue
        if not instance_legal:
            print('Instance format illegal before index %d' % idx)
            idx -= 1
            continue

        st += st2 + '\n'
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


def csv_extracting(log_file, col_header_file, label_file, dataset='HDFS'):
    '''
    :param log_file: txt/log
    :param col_header_file: txt
    :param label_file: csv
    :return: None
    '''
    assert log_file.endswith('.txt') or log_file.endswith('.log')
    if dataset == 'HDFS':
        print("== STEP 1 ==")
        dataloader.load_HDFS(log_file, col_header_file, label_file)
    else:
        print("== STEP 1 ==")
        _load_openstack(log_file, col_header_file)

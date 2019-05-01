# STEP 7

import pandas as pd
import pickle
from loglizer.vd_workflow.valid_template import generate_filenames


def _generate_mean_std(file):
    '''
    :param file:
    write a list with (n + 1) elements in pickle format, where n is the number of args,
        each element is a tuple (mean, std)
    '''
    df = pd.read_csv(file)
    result = [(df['DeltaTime'].mean(), df['DeltaTime'].std())]
    for i in range(4):
        col = 'Arg%d' % i
        if df[col].isnull().any():
            break
        result.append((df[col].mean(), df[col].std()))
    with open(file[: -4] + '.pkl', 'wb') as f:
        pickle.dump(result, f)


def mean_std_stat(train_file, valid_template, dataset='HDFS'):
    if dataset == 'HDFS':
        print('== STEP 7 ==')
    else:
        print('== STEP 5 ==')
    for file in generate_filenames(train_file, valid_template):
        _generate_mean_std(file)


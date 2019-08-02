# STEP 2

import sys
import pandas as pd
import re

label_dict = None


def _extract_block(str):
    try:
        return list(set(re.findall(r'(blk_-?\d+)', str)))
    except Exception:
        return []


def _detect_anomaly(blockIds):
    assert label_dict
    anomaly_result = [label_dict[t] for t in blockIds]
    return 1 if 'Anomaly' in anomaly_result else 0


def load_label_file(label_file):
    global label_dict
    label_dict = pd.read_csv(label_file).set_index('BlockId')['Label'].to_dict()


def csv_block_extracting(f, t):
    '''
    :param f: from
    :param t: to
    :return:
    '''
    print("== STEP 2 ==")
    df = pd.read_csv(f)
    df.drop(['Component', 'EventTemplate'], axis=1, inplace=True)
    df['BlockIds'] = df['Content'].apply(_extract_block)
    df.drop(['Content'], axis=1, inplace=True)
    df['Tag'] = df['BlockIds'].apply(_detect_anomaly)
    df.to_csv(t, index=False)

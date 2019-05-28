import pandas as pd
import numpy as np


def _merge_decimal(row, i_part, d_part):
    return float(row[i_part] + '.' + row[d_part])


def _merge(filename, i_col, d_col):
    df = pd.read_csv(filename, dtype=str)
    df[i_col] = df.apply(_merge_decimal, axis=1, args=(i_col, d_col))
    k = int(d_col[-1])
    while k < 3:
        df['Arg%d' % k] = df['Arg%d' % (k + 1)]
        k += 1
    df['Arg3'] = np.nan
    df.to_csv(filename, index=False)


def integrate_decimal(train_file, validate_file, test_file, valid_templates):
    print('== STEP 4 ==')
    templates_to_merge = [('E10', 0, 1),
                          ('E10', 1, 2),
                          ('E12', 0, 1),
                          ('E13', 0, 1),
                          ('E14', 0, 1),
                          ('E15', 0, 1),
                          ('E16', 1, 2),
                          ('E17', 1, 2),
                          ('E18', 1, 2)]
    for file in [train_file, validate_file, test_file]:
        for ttm in templates_to_merge:
            if ttm[0] in valid_templates:
                filename = file[: -4] + '_' + ttm[0] + '.csv'
                _merge(filename, 'Arg%d' % ttm[1], 'Arg%d' % ttm[2])

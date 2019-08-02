# STEP 4

import pandas as pd


def _classify_file(filename, template_list):
    df = pd.read_csv(filename, dtype=str)
    for template in template_list:
        df_t = df.ix[df['EventId'] == template]
        df_t.to_csv(filename[: -4] + '_' + template + '.csv', index=False)


def _template_count(filename):
    df = pd.read_csv(filename, dtype=str)
    return df['EventId'].value_counts().to_dict()


def classify(train_file, validate_file, test_file, tot_syms, train_min, validate_min, test_min, dataset='HDFS'):
    if dataset == 'HDFS':
        print("== STEP 4 ==")
    else:
        print("== STEP 2 ==")
    train_tc = _template_count(train_file)
    validate_tc = _template_count(validate_file)
    test_tc = _template_count(test_file)

    valid_templates = []
    for i in range(tot_syms):
        template = 'E%d' % i
        if template in train_tc.keys() and train_tc[template] >= train_min \
            and template in validate_tc.keys() and validate_tc[template] >= validate_min \
            and template in test_tc.keys() and test_tc[template] >= test_min:
            valid_templates.append(template)

    _classify_file(train_file, valid_templates)
    _classify_file(validate_file, valid_templates)
    _classify_file(test_file, valid_templates)

    return valid_templates

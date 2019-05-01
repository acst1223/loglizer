#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
sys.path.append('/root/')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing
from scripts import config

config.init('PCA_OpenStack')
data_train = config.OpenStack_train_data
data_test = config.OpenStack_test_data


if __name__ == '__main__':
    (x_train, y_train), (_, _), (_, _) = dataloader.load_OpenStack(data_train,
                                                                train_ratio=1,
                                                                split_type='uniform',
                                                                is_data_instance=True)
    (_, _), (x_test, y_test), (_, _) = dataloader.load_OpenStack(data_test,
                                                                   train_ratio=0,
                                                                   split_type='uniform',
                                                                   is_data_instance=True)

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
                                              normalization='zero-mean')
    x_test = feature_extractor.transform(x_test)

    model = PCA()
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

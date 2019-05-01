#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
sys.path.append('/path/')
from loglizer.models import InvariantsMiner
from loglizer import dataloader, preprocessing
from scripts import config

config.init('InvariantsMiner_HDFS')
data_instances = config.HDFS_data
epsilon = 0.5

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), (_, _) = dataloader.load_HDFS(data_instances,
                                                                train_ratio=0.3,
                                                                split_type='uniform',
                                                                test_ratio=0.6,
                                                                is_data_instance=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)


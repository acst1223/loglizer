#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
from loglizer.models import LogClustering
from loglizer import preprocessing
from scripts import config
from loglizer.BGL_workflow.data_generator import generate_data

config.init('LogClustering_BGL')
data_instances = config.BGL_data
max_dist = 0.3  # the threshold to stop the clustering process
anomaly_threshold = 0.3  # the threshold for anomaly detection


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), (_, _) = generate_data(data_instances, 0.3, 0.6)

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

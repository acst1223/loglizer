#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
from loglizer.models import LogClustering
from loglizer import preprocessing
from workflow.BGL_workflow.data_generator import load_BGL
from workflow import dataloader
import config

datasets = ['HDFS', 'BGL']

max_dist = 0.3  # the threshold to stop the clustering process
anomaly_threshold = 0.3  # the threshold for anomaly detection

if __name__ == '__main__':
    for dataset in datasets:
        print('########### Start Log Cluster on Dataset ' + dataset + ' ###########')
        config.init('LogClustering_' + dataset)

        if dataset == 'BGL':

            data_instances = config.BGL_data
            (x_train, y_train), (x_test, y_test), (_, _) = load_BGL(data_instances, 0.3, 0.6)

        if dataset == 'HDFS':
            data_instances = config.HDFS_data
            (x_train, y_train), (x_test, y_test), (_, _) = dataloader.load_HDFS(data_instances,
                                                                                train_ratio=0.3,
                                                                                split_type='uniform',
                                                                                test_ratio=0.6,
                                                                                is_data_instance=True)

        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
        x_test = feature_extractor.transform(x_test)

        model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
        model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

        print('Train validation:')
        precision, recall, f1 = model.evaluate(x_train, y_train)

        print('Test validation:')
        precision, recall, f1 = model.evaluate(x_test, y_test)

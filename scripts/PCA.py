import sys

sys.path.append('../')
sys.path.append('/root/')
from loglizer.models import PCA
from loglizer import preprocessing, dataloader
from workflow.BGL_workflow.data_generator import load_BGL
import config

datasets = ['BGL','HDFS', 'OpenStack']



if __name__ == '__main__':
    for dataset in datasets:
        print('########### Start Invariant Mining on Dataset ' + dataset + ' ###########')
        config.init('PCA_' + dataset)

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
        if dataset == 'OpenStack':
            data_train = config.OpenStack_train_data
            data_test = config.OpenStack_test_data
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

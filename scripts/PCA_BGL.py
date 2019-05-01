import sys

sys.path.append('../')
sys.path.append('/root/')
from loglizer.models import PCA
from loglizer.BGL_workflow.data_generator import generate_data
from loglizer import preprocessing
from scripts import config

config.init('PCA_BGL')
data_instances = config.BGL_data


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), (_, _) = generate_data(data_instances, 0.3, 0.6)

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

import sys

sys.path.append('../')
sys.path.append('/path/')
from loglizer.models import InvariantsMiner
from loglizer import preprocessing
from loglizer.BGL_workflow.data_generator import generate_data
from scripts import config

config.init('InvariantsMiner_BGL')
epsilon = 0.5
data_instances = config.BGL_data


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), (_, _) = generate_data(data_instances, 0.3, 0.6)

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    model = InvariantsMiner(epsilon=epsilon)
    model.fit(x_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

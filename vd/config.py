import tensorflow as tf
import os
import sys

path = ''
if os.path.exists('../vd'):
    print('Config: PyCharm mode')
    path = '../'
else:
    print('Config: Docker mode')
    path = '/root/'

HDFS_data = path + 'data/HDFS/data_instances.csv'
HDFS_col_header = path + 'data/HDFS/col_header.csv'
OpenStack_vd_data = path + 'data/OpenStack/openstack'
OpenStack_train_data = path + 'data/OpenStack/openstack_val_normal1_instances.csv'
OpenStack_test_data = path + 'data/OpenStack/openstack_val_with_performance_anomalies_instances.csv'
OpenStack_valid_template = path + 'data/OpenStack/valid_template.pkl'
OpenStack_result_png_prefix = path + 'data/OpenStack/result'
log_path = ''



def init(model_name):
    global log_path
    log_path = path + 'log/' + model_name + '.log'


def log(str):
    print(str)
    f = open(log_path, 'a')
    f.write('%s\n' % str)
    f.close()


def get_stat_path(data_path):
    return data_path[: -4] + '.stat.pkl'


def get_OpenStack_result_png_name(template):
    return OpenStack_result_png_prefix + '_' + template + '.png'

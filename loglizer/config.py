import tensorflow as tf
import os
import sys

path = ''
if os.path.exists('../scripts'):
    print('Config: Normal mode')
    path = '../'
else:
    print('Config: Docker mode')
    path = '/root/'

HDFS_data = path + 'data/HDFS/data_instances.csv'
HDFS_col_header = path + 'data/HDFS/col_header.csv'
BGL_col_header = path + 'data/BGL/BGL2.log_templates.csv'
BGL_data = path + 'data/BGL/event_sequence2.csv'
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

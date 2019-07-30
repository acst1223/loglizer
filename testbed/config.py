import tensorflow as tf
import os
import sys

path = ''
if os.path.exists('../scripts'):
    print('Config: PyCharm mode')
    path = '../'
else:
    print('Config: Docker mode')
    path = '/root/'

testbed_path = path + 'data/testbed/'
log_path = ''


def init(model_name):
    global log_path
    log_path = path + 'log/' + model_name + '.log'


def log(st):
    print(st)
    f = open(log_path, 'a')
    f.write('%s\n' % st)
    f.close()

import pandas as pd
from loglizer.dataloader import load_OpenStack

# data_df = pd.read_csv('data/HDFS/data_instances.csv').sample(frac=1)
# print('session 2')
# data_df.to_csv('data/HDFS/data_instances_another.csv', index=False)
log_file = 'data/OpenStack/openstack_val_normal2.csv'
label_file = 'data/OpenStack/anomalous_instance'
col_header_file = '../../data/openDataSet/logs/OpenStack/OpenStack_2k.log_templates.csv'
# load_OpenStack(log_file, col_header_file, save_csv=True)
load_OpenStack(log_file, label_file=label_file, save_csv=True)

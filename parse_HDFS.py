from loglizer.dataloader import load_HDFS
from loglizer.preprocessing import gen_stat
import pandas as pd

HDFS_file = '../../data/openDataSet/logs/HDFS/HDFS.log'
header_file = '../../data/openDataSet/logs/HDFS/col_header.txt'
label_file = '../../data/openDataSet/logs/HDFS/label.csv'
csv_file = HDFS_file[: -4] + '.csv'
instance_file = 'data/HDFS/data_instances.csv'

# load_HDFS(HDFS_file, col_header_file=header_file)
# load_HDFS(csv_file, label_file=label_file, save_csv=True, extract_pids=True, extract_time_series=True)
# data_df = pd.read_csv('data/HDFS/data_instances.csv').sample(frac=1)
# data_df.to_csv('data/HDFS/data_instances.csv', index=False)
gen_stat(instance_file)

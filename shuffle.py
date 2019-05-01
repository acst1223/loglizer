import pandas as pd

data_df = pd.read_csv('data/HDFS/data_instances.csv').sample(frac=1)
data_df.to_csv('data/HDFS/data_instances_another.csv', index=False)
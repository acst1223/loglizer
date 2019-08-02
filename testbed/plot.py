from testbed import config
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime


h = 10
avg_n = 200


if __name__ == '__main__':
    top_counts_file_name = config.path + 'top_counts.pkl'
    with open(top_counts_file_name, 'rb') as f:
        top_counts = pickle.load(f)

    file = config.testbed_path + 'logstash-2019.07.22_ts-food-service_sorted.csv_structured.csv'
    df = pd.read_csv(file).iloc[h:]

    datetime_original_series = pd.Series(df['DateTime'].values)
    top_counts_original_series = pd.Series(top_counts)
    datetime_series = []
    top_counts_series = []

    for i in range(len(datetime_original_series) // avg_n):
        datetime_series.append(datetime_original_series[i * avg_n])
        top_counts_series.append(top_counts_original_series[i * avg_n: (i + 1) * avg_n].mean())
    datetime_series = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%S+00:00') for t in datetime_series]

    plot_df = pd.DataFrame()
    plot_df['DateTime'] = datetime_series
    plot_df['top_counts'] = top_counts_series
    plot_df.set_index('DateTime', inplace=True)

    ax = plt.subplot(1, 1, 1)
    plt.xlabel('Time')
    plt.plot(datetime_series, top_counts_series, '-')
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
    plt.show()

# STEP 6

import pandas as pd
import datetime


def _to_date(date, dataset):
    if dataset == 'HDFS':
        date = '20%06d' % date
        return '%s/%s/%s' % (date[0: 4], date[4: 6], date[6: 8])
    else:
        return date


def _to_time(time, dataset):
    if dataset == 'HDFS':
        time = '%06d' % time
        return '%s:%s:%s' % (time[0: 2], time[2: 4], time[4: 6])
    else:
        return time


def _time_interval(row, time1, time2, dataset):
    if dataset == 'HDFS':
        time1 = datetime.datetime.strptime(row[time1], '%Y/%m/%d %H:%M:%S')
        time2 = datetime.datetime.strptime(row[time2], '%Y/%m/%d %H:%M:%S')
        return (time2 - time1).seconds
    else:
        time1 = datetime.datetime.strptime(row[time1], '%Y-%m-%d %H:%M:%S.%f')
        time2 = datetime.datetime.strptime(row[time2], '%Y-%m-%d %H:%M:%S.%f')
        return int((time2 - time1).microseconds / 1000)


def _transform(filename, dataset):
    df = pd.read_csv(filename, dtype=str)
    df['Date2'] = df['Date'].apply(_to_date, dataset=dataset)
    df['Time2'] = df['Time'].apply(_to_time, dataset=dataset)
    df['Time2'] = df['Date2'] + ' ' + df['Time2']
    df['Date2'] = df['Time2'].shift()
    df.loc[0, 'Date2'] = df.loc[0, 'Time2']
    df['DeltaTime'] = df.apply(_time_interval, axis=1, args=('Date2', 'Time2', dataset))
    df.drop(['Date2', 'Time2'], axis=1, inplace=True)
    df.to_csv(filename, index=False)


def transform(train_file, validate_file, test_file, valid_templates, dataset='HDFS'):
    if dataset == 'HDFS':
        print("== STEP 6 ==")
    else:
        print("== STEP 3 ==")

    for template in valid_templates:
        filename = train_file[: -4] + '_' + template + '.csv'
        _transform(filename, dataset)
        filename = validate_file[: -4] + '_' + template + '.csv'
        _transform(filename, dataset)
        filename = test_file[: -4] + '_' + template + '.csv'
        _transform(filename, dataset)

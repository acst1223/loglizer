import pandas as pd


def drop_csv(input_file, output_file):
    '''
    drop unnessary columns and process labels to 0(Normal)/1(Anomaly)
    '''
    print("== STEP 2 ==")
    df = pd.read_csv(input_file)
    df.drop(['Timestamp', 'Date', 'NodeRepeat', 'Type', 'Component',
             'Level', 'Content', 'EventTemplate'], axis=1, inplace=True)
    df['Label'] = df['Label'].apply(lambda t: 0 if t == '-' else 1)
    df.to_csv(output_file, index=False)

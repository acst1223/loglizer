import pandas as pd


def generate_event_sequence(slim_csv_file, output_file):
    print("== STEP 3 ==")
    data_dict = {}
    label_dict = {}
    df = pd.read_csv(slim_csv_file)
    for idx, row in df.iterrows():
        if idx % 100000 == 0:
            print("%d rows processed" % idx)
        node = row['Node']
        if not node in data_dict:
            data_dict[node] = []
            label_dict[node] = 0
        data_dict[node].append(row['EventId'])
        label_dict[node] |= row['Label']
    data_df = pd.DataFrame(list(data_dict.items()), columns=['Node', 'EventSequence'])
    label_df = pd.DataFrame(list(label_dict.items()), columns=['Node', 'Label'])
    df = pd.merge(data_df, label_df, on='Node').sample(frac=1)
    df.to_csv(output_file, index=False)

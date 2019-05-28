# STEP 3

import pandas as pd


def split(extracted_csv, train_target, validate_target, test_target, train_ratio, test_ratio):
    print("== STEP 3 ==")
    df = pd.read_csv(extracted_csv)
    l = len(df)
    train_point = int(l * train_ratio)
    test_point = -int(l * test_ratio)
    train_df = df[: train_point]
    validate_df = df[train_point: test_point]
    test_df = df[test_point:]
    train_df = train_df.ix[df['Tag'] == 0]  # throw away exceptions in training set
    train_df.to_csv(train_target, index=False)
    validate_df.to_csv(validate_target, index=False)
    test_df.to_csv(test_target, index=False)

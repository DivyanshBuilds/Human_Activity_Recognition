import pandas as pd
from src.data_ingestion import ingest_data

EXPECTED_COLUMNS = 563
MISSING_VALUE_THRESHOLD = 0.05   # 5%
DUPLICATE_THRESHOLD = 0.05       # 5%

def validate_data(train_df, test_df):
    errors = []
    warnings = []

    # column count check
    if train_df.shape[1] != EXPECTED_COLUMNS:
        errors.append(f"Train column mismatch. Expected {EXPECTED_COLUMNS}, got {train_df.shape[1]}")
    if test_df.shape[1] != EXPECTED_COLUMNS:
        errors.append(f"Test column mismatch. Expected {EXPECTED_COLUMNS}, got {test_df.shape[1]}")

    # missing values check
    for name, df in [('train', train_df), ('test', test_df)]:
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > MISSING_VALUE_THRESHOLD:
            errors.append(f"{name} has too many missing values: {missing_ratio:.2%}")
        elif missing_ratio > 0:
            warnings.append(f"{name} has some missing values: {missing_ratio:.2%}")

    # duplicates check
    for name, df in [('train', train_df), ('test', test_df)]:
        duplicate_ratio = df.duplicated().sum() / len(df)
        if duplicate_ratio > DUPLICATE_THRESHOLD:
            errors.append(f"{name} has too many duplicates: {duplicate_ratio:.2%}")
        elif duplicate_ratio > 0:
            warnings.append(f"{name} has some duplicates: {duplicate_ratio:.2%}")

    # activity labels check
    expected_activities = {1, 2, 3, 4, 5, 6}
    if set(train_df['activity'].unique()) != expected_activities:
        errors.append(f"Unexpected activity labels in train: {set(train_df['activity'].unique())}")
    if set(test_df['activity'].unique()) != expected_activities:
        errors.append(f"Unexpected activity labels in test: {set(test_df['activity'].unique())}")

    # print results
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        raise ValueError("Data validation failed. Fix errors before proceeding.")

    print("Data validation passed successfully.")


if __name__ == '__main__':
    train_df, test_df = ingest_data()
    validate_data(train_df, test_df)
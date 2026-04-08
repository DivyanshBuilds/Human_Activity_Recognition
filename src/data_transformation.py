import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

from src.data_ingestion import ingest_data
from src.data_validation import validate_data


def transform_data(train_df, test_df):
    # separate X and y
    X_train = train_df.drop(['subject', 'activity'], axis=1)
    y_train = train_df['activity']

    X_test = test_df.drop(['subject', 'activity'], axis=1)
    y_test = test_df['activity']

    # scale features - fit only on train, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # save scaler
    os.makedirs(r'C:\Users\Divyansh\Desktop\har_project\models', exist_ok=True)
    scaler_path = r'C:\Users\Divyansh\Desktop\har_project\models\scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to models/scaler.pkl")

    # convert back to dataframe with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # reset index
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # save to data/processed
    os.makedirs(r'C:\Users\Divyansh\Desktop\har_project\data\processed', exist_ok=True)

    X_train_scaled.to_csv(r'C:\Users\Divyansh\Desktop\har_project\data\processed\X_train.csv', index=False)
    X_test_scaled.to_csv(r'C:\Users\Divyansh\Desktop\har_project\data\processed\X_test.csv', index=False)
    y_train.to_csv(r'C:\Users\Divyansh\Desktop\har_project\data\processed\y_train.csv', index=False)
    y_test.to_csv(r'C:\Users\Divyansh\Desktop\har_project\data\processed\y_test.csv', index=False)

    print("Data transformation complete. Files saved to data/processed.")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    train_df, test_df = ingest_data()
    validate_data(train_df, test_df)
    X_train, X_test, y_train, y_test = transform_data(train_df, test_df)
    print(X_train.shape)
    print(X_test.shape)
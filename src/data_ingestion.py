import pandas as pd

def ingest_data():
    base_path = r'C:\Users\Divyansh\Desktop\har_project\data\raw'
    
    # load feature names
    features = pd.read_csv(f'{base_path}\\features.txt', sep='\s+', header=None)
    feature_names = features[1].tolist()
    
    # load train data
    X_train = pd.read_csv(f'{base_path}\\X_train.txt', sep='\s+', header=None)
    y_train = pd.read_csv(f'{base_path}\\y_train.txt', sep='\s+', header=None)
    subject_train = pd.read_csv(f'{base_path}\\subject_train.txt', sep='\s+', header=None)
    
    X_train.columns = feature_names
    y_train.columns = ['activity']
    subject_train.columns = ['subject']
    
    train_df = pd.concat([subject_train, X_train, y_train], axis=1)
    
    # load test data
    X_test = pd.read_csv(f'{base_path}\\X_test.txt', sep='\s+', header=None)
    y_test = pd.read_csv(f'{base_path}\\y_test.txt', sep='\s+', header=None)
    subject_test = pd.read_csv(f'{base_path}\\subject_test.txt', sep='\s+', header=None)
    
    X_test.columns = feature_names
    y_test.columns = ['activity']
    subject_test.columns = ['subject']
    
    test_df = pd.concat([subject_test, X_test, y_test], axis=1)
    
    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = ingest_data()
    print(train_df.shape)
    print(test_df.shape)
import pandas as pd
import pickle
import numpy as np

with open(r'C:\Users\Divyansh\Desktop\har_project\models\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(r'C:\Users\Divyansh\Desktop\har_project\models\best_model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv(r'C:\Users\Divyansh\Desktop\har_project\data\processed\X_test.csv')
y_test = pd.read_csv(r'C:\Users\Divyansh\Desktop\har_project\data\processed\y_test.csv').squeeze()

sample_idx = np.random.randint(0, len(X_test))
sample = X_test.iloc[[sample_idx]]
actual = y_test.iloc[sample_idx]

prediction = model.predict(sample.values)

activity_map = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

print(f"Sample index: {sample_idx}")
print(f"Actual activity:    {actual} — {activity_map[actual]}")
print(f"Predicted activity: {prediction[0]} — {activity_map[prediction[0]]}")
print(f"Correct: {actual == prediction[0]}")
import pandas as pd
import pickle
import numpy as np
import os
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model and scaler
with open(os.path.join(BASE_DIR, 'models', 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# load test data
X_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_test.csv'))
y_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'y_test.csv')).squeeze()

# pick 30 random samples
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), 30, replace=False).tolist()

activity_map = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

top_10_features = [
    'tBodyAcc-mean()-X',
    'tBodyAcc-mean()-Y',
    'tBodyAcc-mean()-Z',
    'tBodyAcc-std()-X',
    'tBodyAcc-std()-Y',
    'tBodyAcc-std()-Z',
    'tGravityAcc-mean()-X',
    'tGravityAcc-mean()-Y',
    'tGravityAcc-mean()-Z',
    'tBodyAccMag-mean()'
]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    actual = None
    selected_index = None
    features = None
    correct = None

    if request.method == 'POST':
        selected_index = int(request.form['sample_index'])
        sample = X_test.iloc[[selected_index]]
        actual_label = int(y_test.iloc[selected_index])
        predicted_label = int(model.predict(sample.values)[0])

        actual = f"{actual_label} — {activity_map[actual_label]}"
        prediction = f"{predicted_label} — {activity_map[predicted_label]}"
        correct = actual_label == predicted_label

        features = {col: round(float(sample[col].values[0]), 4) for col in top_10_features}

    return render_template('index.html',
                           sample_indices=sample_indices,
                           selected_index=selected_index,
                           actual=actual,
                           prediction=prediction,
                           correct=correct,
                           features=features)


if __name__ == '__main__':
    app.run(debug=True)
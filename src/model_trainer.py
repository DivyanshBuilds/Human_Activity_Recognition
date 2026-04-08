import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_ingestion import ingest_data
from src.data_validation import validate_data
from src.data_transformation import transform_data


def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma=0.001, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"{name} Accuracy: {acc:.4f}")
        print(f"Classification Report:\n{results[name]['report']}")
        print(f"Confusion Matrix:\n{results[name]['confusion_matrix']}")

    return results


def save_best_model(results):
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_name]['model']

    os.makedirs(r'C:\Users\Divyansh\Desktop\har_project\models', exist_ok=True)
    pkl_path = r'C:\Users\Divyansh\Desktop\har_project\models\best_model.pkl'

    with open(pkl_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\nBest model: {best_name} with accuracy {results[best_name]['accuracy']:.4f}")
    print(f"Model saved to {pkl_path}")

    return best_name, pkl_path


if __name__ == '__main__':
    train_df, test_df = ingest_data()
    validate_data(train_df, test_df)
    X_train, X_test, y_train, y_test = transform_data(train_df, test_df)
    results = train_models(X_train, X_test, y_train, y_test)
    save_best_model(results)
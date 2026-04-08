# Human Activity Recognition Using Smartphones

## Problem Statement
The goal of this project is to classify human physical activities — Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying — using sensor data recorded from a smartphone worn on the waist. This is a multi-class classification problem with 6 activity classes.

---

## Dataset
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

- 30 volunteers aged 19-48 years
- Samsung Galaxy S II worn on the waist
- Accelerometer and gyroscope sensors recorded at 50Hz
- Data segmented into 2.56 second windows (128 readings per window)
- 561 features extracted from time and frequency domain signals
- 10,299 total samples — 7,352 training, 2,947 test
- Train/test split done by person — 21 people for training, 9 for testing

---

## Solution Approach
1. Load raw sensor data from txt files
2. Validate data quality — missing values, duplicates, schema checks
3. Apply StandardScaler for feature scaling
4. Train multiple ML models and compare performance
5. Use GridSearchCV to tune best model hyperparameters
6. Save best model and scaler as pkl files for prediction

---

## Project Structure
```
har_project/
├── data/
│   ├── raw/                    # raw txt files from UCI
│   └── processed/              # scaled csv files
├── models/
│   ├── best_model.pkl          # saved best model
│   └── scaler.pkl              # saved StandardScaler
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py       # loads raw data
│   ├── data_validation.py      # validates data quality
│   ├── data_transformation.py  # scales and saves processed data
│   └── model_trainer.py        # trains and evaluates all models
├── eda.ipynb                    # EDA and experimentation notebook
├── main.py                     # orchestrates full pipeline
├── predict.py                  # loads model and makes predictions
└── requirements.txt
```

---

## Requirements
Install all dependencies using:

```
pip install -r requirements.txt
```

---

## How to Run

**Full pipeline (ingestion → validation → transformation → training):**
```
python main.py
```

**Prediction on a random test sample:**
```
python predict.py
```

---

## Key EDA Findings
- Dataset is well balanced — activity classes range from 13% to 19% of total data
- t-SNE visualization revealed two natural clusters in the data:
  - **Dynamic activities** (Walking, Walking Upstairs, Walking Downstairs) — well separated from static
  - **Static activities** (Sitting, Standing, Laying) — Sitting and Standing heavily overlap
- LAYING is the easiest activity to classify — completely isolated in t-SNE
- Sitting vs Standing is the hardest pair — sensor readings are nearly identical for both
- Features are already normalised to [-1, 1] by dataset authors — StandardScaler applied additionally for zero-centring

---

## Model Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 95.49% |
| SVM (tuned) | 95.52% |
| XGBoost | 93.82% |
| Random Forest | 92.60% |
| KNN | 88.02% |

---

## Best Model & Hyperparameters
**Model:** Support Vector Machine (SVM) with RBF kernel

**Hyperparameters tuned using GridSearchCV (5-fold cross validation):**
- `C = 10`
- `gamma = 0.001`
- `kernel = rbf`

**Test Accuracy: 95.52%**

The model performs well across all activities except Sitting vs Standing — which is a fundamental limitation of the sensor data itself, not the model.

---

## Acknowledgements
- Dataset: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones. UCI Machine Learning Repository.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu)
- [Scikit-learn](https://scikit-learn.org)
- [XGBoost](https://xgboost.readthedocs.io)

---

## Built By
**Divyansh**
- This project was built as a practical learning exercise after completing a Machine Learning course, applying end-to-end ML pipeline concepts on a real world dataset.
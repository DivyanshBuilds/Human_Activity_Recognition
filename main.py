from src.data_ingestion import ingest_data
from src.data_validation import validate_data
from src.data_transformation import transform_data
from src.model_trainer import train_models, save_best_model


def main():
    print("="*50)
    print("Step 1: Data Ingestion")
    print("="*50)
    train_df, test_df = ingest_data()
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\n" + "="*50)
    print("Step 2: Data Validation")
    print("="*50)
    validate_data(train_df, test_df)

    print("\n" + "="*50)
    print("Step 3: Data Transformation")
    print("="*50)
    X_train, X_test, y_train, y_test = transform_data(train_df, test_df)

    print("\n" + "="*50)
    print("Step 4: Model Training")
    print("="*50)
    results = train_models(X_train, X_test, y_train, y_test)

    print("\n" + "="*50)
    print("Step 5: Saving Best Model")
    print("="*50)
    best_name, pkl_path = save_best_model(results)

    print("\n" + "="*50)
    print("Pipeline Complete!")
    print(f"Best Model: {best_name}")
    print(f"Saved at: {pkl_path}")
    print("="*50)


if __name__ == '__main__':
    main()
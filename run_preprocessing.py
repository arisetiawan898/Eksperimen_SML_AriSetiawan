import os
import pandas as pd
from preprocessing.automate_AriSetiawan import preprocess_data


if __name__ == "__main__":
    raw_path = "insurance_raw.csv"
    save_pipeline_path = "preprocessing/preprocessor.joblib"
    save_header_path = "preprocessing/insurance_preprocessing/columns.csv"
    save_dataset_path = "preprocessing/insurance_preprocessing"

    os.makedirs(save_dataset_path, exist_ok=True)

    # Load data raw
    print("Membaca dataset...")
    df = pd.read_csv(raw_path)
    print(f"Dataset berhasil dibaca. Shape: {df.shape}")

    # Jalankan preprocessing
    print("Menjalankan preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(
        data=df,
        target_column="charges",
        save_path=save_pipeline_path,
        file_path=save_header_path
    )

    # Simpan hasil ke CSV
    train_df = pd.DataFrame(X_train).assign(target=y_train.reset_index(drop=True))
    test_df = pd.DataFrame(X_test).assign(target=y_test.reset_index(drop=True))

    train_df.to_csv(f"{save_dataset_path}/insurance_train_preprocessed.csv", index=False)
    test_df.to_csv(f"{save_dataset_path}/insurance_test_preprocessed.csv", index=False)

    print(f"Train set: {train_df.shape}")
    print(f"Test set:  {test_df.shape}")
    print("Preprocessing selesai!")

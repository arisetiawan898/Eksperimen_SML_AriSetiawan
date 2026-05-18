from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import os


def preprocess_data(data, target_column, save_path, file_path):
    """
    Melakukan preprocessing data insurance secara otomatis.
    
    Args:
        data: DataFrame input
        target_column: nama kolom target
        save_path: path untuk menyimpan pipeline (.joblib)
        file_path: path untuk menyimpan header kolom (.csv)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Identifikasi fitur numerik dan kategorikal
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Simpan nama kolom (tanpa target)
    column_names = data.columns.drop(target_column)
    df_header = pd.DataFrame(columns=column_names)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom disimpan ke: {file_path}")

    # Hapus target dari fitur
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Definisi transformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit dan transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Simpan pipeline
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(preprocessor, save_path)
    print(f"Pipeline preprocessing disimpan ke: {save_path}")

    return X_train, X_test, y_train, y_test

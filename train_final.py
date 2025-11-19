import os
import re
import argparse
import sys

# required imports may not be available in every environment; import with friendly error messages
try:
    import numpy as np
    import pandas as pd
    import joblib
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"{e}. Install project dependencies with: python -m pip install -r requirements_final.txt") from e

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"{e}. Install scikit-learn in your environment (see requirements_final.txt)") from e


def load_data(csv_path: str = None, use_kaggle: bool = True) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if use_kaggle:
        try:
            import kagglehub
            dataset_dir = kagglehub.dataset_download("taeefnajib/used-car-price-prediction-dataset")
            candidate = os.path.join(dataset_dir, "used_cars.csv")
            if os.path.exists(candidate):
                return pd.read_csv(candidate)
        except Exception:
            pass
    fallback = os.path.join(os.getcwd(), "used_cars.csv")
    if os.path.exists(fallback):
        return pd.read_csv(fallback)
    raise FileNotFoundError("used_cars.csv not found; provide --csv or enable kagglehub")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # accept both common spellings 'mileage' and the typo 'milage'
    col_map_actual = {'model_year': 'year', 'milage': 'mileage_km', 'mileage': 'mileage_km', 'fuel_type': 'fuel'}
    df = df.rename(columns=col_map_actual)

    if 'price' in df.columns:
        df['price'] = df['price'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])

    if 'mileage_km' in df.columns:
        df['mileage_km'] = df['mileage_km'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['mileage_km'] = pd.to_numeric(df['mileage_km'], errors='coerce')

    if 'engine' in df.columns:
        df['engine_cc'] = df['engine'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
        df['engine_cc'] = pd.to_numeric(df['engine_cc'], errors='coerce')
        df = df.drop(columns=['engine'], errors='ignore')

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['car_age'] = pd.Timestamp.now().year - df['year']

    # Drop columns that won't be used
    df = df.drop(columns=[c for c in ['name', 'year'] if c in df.columns], errors='ignore')

    return df


def build_pipeline(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # numeric pipeline: median impute + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical pipeline: impute missing + one-hot encode
    # Use `sparse=False` for OneHotEncoder to maintain compatibility across scikit-learn versions
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def train_and_save(csv: str, out_path: str, no_kaggle: bool, test_size: float = 0.2, random_state: int = 42):
    df = load_data(csv_path=csv, use_kaggle=not no_kaggle)
    df = preprocess_df(df)

    if 'price' not in df.columns:
        raise ValueError('price column not found after preprocessing')

    X = df.drop('price', axis=1)
    y = df['price']

    # One-hot encode any remaining object columns to keep pipeline simple
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(obj_cols) > 0:
        # Leave encoding to pipeline, but ensure columns exist for ColumnTransformer
        pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = build_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print('Training completed.')
    print(f'R-squared: {r2:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    joblib.dump(pipeline, out_path)
    print(f'Model pipeline saved to: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Train final car price prediction model and save pipeline')
    parser.add_argument('--csv', type=str, default=None, help='Path to used_cars.csv')
    parser.add_argument('--out', type=str, default='car_price_pipeline.joblib', help='Output path for saved pipeline')
    parser.add_argument('--no-kaggle', action='store_true', help='Do not attempt to download via kagglehub')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    args = parser.parse_args()

    train_and_save(csv=args.csv, out_path=args.out, no_kaggle=args.no_kaggle, test_size=args.test_size, random_state=args.random_state)


if __name__ == '__main__':
    main()

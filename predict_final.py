import argparse
import os
import re

# friendly imports with actionable error messages
try:
    import pandas as pd
    import joblib
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"{e}. Install project dependencies with: python -m pip install -r requirements_final.txt") from e


def predict(csv_in: str, model_path: str, csv_out: str = None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    pipeline = joblib.load(model_path)
    df = pd.read_csv(csv_in)
    # Apply the same preprocessing used during training so pipeline columns match
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

        df = df.drop(columns=[c for c in ['name', 'year'] if c in df.columns], errors='ignore')
        return df

    X = preprocess_df(df.copy())
    preds = pipeline.predict(X)
    df['predicted_price'] = preds
    out = csv_out or os.path.splitext(csv_in)[0] + '_predictions.csv'
    df.to_csv(out, index=False)
    print(f'Predictions saved to: {out}')


def main():
    parser = argparse.ArgumentParser(description='Make predictions with saved car price pipeline')
    parser.add_argument('--csv', required=True, help='Input CSV with features (no price required)')
    parser.add_argument('--model', default='car_price_pipeline.joblib', help='Path to saved pipeline')
    parser.add_argument('--out', default=None, help='Path to output CSV with predictions')
    args = parser.parse_args()
    predict(args.csv, args.model, args.out)


if __name__ == '__main__':
    main()

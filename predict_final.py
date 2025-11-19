import argparse
import os
import re
import pandas as pd
import joblib
import numpy as np
import warnings

# silence sklearn version mismatch warnings (optional)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


def numeric_extract(val):
    """Extract digits/decimals from any messy string."""
    if pd.isna(val):
        return None
    val = str(val)
    val = val.replace(",", "")  # remove commas
    match = re.search(r"(\d+\.?\d*)", val)
    return float(match.group(1)) if match else None


def preprocess_df(df: pd.DataFrame, pipeline=None):
    """Mirror training-time preprocessing.

    If a fitted pipeline is provided, ensure all original training columns exist.
    Missing numeric columns -> 0, missing categorical -> 'missing'.
    """
    col_map = {
        "model_year": "year",
        "milage": "mileage_km",
        "mileage": "mileage_km",
        "fuel_type": "fuel",
    }
    df = df.rename(columns=col_map)

    # PRICE (if present)
    if "price" in df.columns:
        df["price"] = df["price"].apply(numeric_extract)
        df = df.dropna(subset=["price"])

    # MILEAGE
    if "mileage_km" in df.columns:
        df["mileage_km"] = df["mileage_km"].apply(numeric_extract)

    # ENGINE
    if "engine" in df.columns:
        df["engine_cc"] = df["engine"].apply(numeric_extract)
        # horsepower extraction patterns
        hp = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)\s*HP", expand=False)
        hp = hp.fillna(df["engine"].astype(str).str.extract(r"^(\d+\.?\d*)", expand=False))
        df["hp"] = pd.to_numeric(hp, errors="coerce")
        df = df.drop(columns=["engine"], errors="ignore")

    # YEAR + CAR AGE
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["car_age"] = pd.Timestamp.now().year - df["year"]

    # Frequency encodings using dataset-level counts (best-effort; if full df not provided may underperform)
    if "brand" in df.columns:
        counts = df["brand"].value_counts()
        df["brand_freq"] = df["brand"].map(counts).fillna(0)
    if "model" in df.columns:
        counts = df["model"].value_counts()
        df["model_freq"] = df["model"].map(counts).fillna(0)

    # Accident / clean title flags
    if "accident" in df.columns:
        df["accident_flag"] = (~df["accident"].astype(str).str.contains("none", case=False, na=False)).astype(int)
    if "clean_title" in df.columns:
        df["clean_title_flag"] = df["clean_title"].astype(str).str.lower().eq("yes").astype(int)

    # drop irrelevant columns
    drop_cols = [c for c in ["name", "year", "brand", "model"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # If pipeline provided, align columns
    if pipeline is not None and hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        try:
            original_cols = list(pipeline.named_steps["preprocess"].feature_names_in_)
            # Determine numeric vs categorical from fitted ColumnTransformer
            ct = pipeline.named_steps["preprocess"]
            numeric_cols = set()
            categorical_cols = set()
            for name, trans, cols in ct.transformers_:
                if name == "num":
                    numeric_cols.update(cols)
                elif name == "cat":
                    categorical_cols.update(cols)
            for col in original_cols:
                if col not in df:
                    if col in numeric_cols:
                        df[col] = 0
                    elif col in categorical_cols:
                        df[col] = "missing"
                    else:
                        df[col] = 0
            # Reorder
            df = df[original_cols]
        except Exception:
            pass

    return df


def predict(csv_in, model_path, csv_out=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load pipeline
    pipeline = joblib.load(model_path)

    df = pd.read_csv(csv_in)
    X = preprocess_df(df.copy(), pipeline=pipeline)

    # Predict with heuristic to detect log-target vs raw-target training
    raw_preds = pipeline.predict(X)
    # If values look like log1p range (<20), invert; else assume already raw prices
    if np.nanmax(raw_preds) < 20:
        df["predicted_price"] = np.expm1(raw_preds)
        scale_msg = "(interpreted as log1p scale; applied expm1)"
    else:
        df["predicted_price"] = raw_preds
        scale_msg = "(interpreted as raw price; no inverse transform)"

    out = csv_out or os.path.splitext(csv_in)[0] + "_predictions.csv"
    df.to_csv(out, index=False)

    print(f"\n✨ Predictions saved to: {out}\n")
    print(df[["predicted_price"]].head())


def main():
    parser = argparse.ArgumentParser(description="Car price prediction tool")
    # Primary required argument
    parser.add_argument("--csv", help="Input CSV file (alias: --input)")
    # Aliases for user convenience
    parser.add_argument("--input", help="Alias for --csv input file")
    parser.add_argument("--model", default="car_price_pipeline.joblib", help="Saved pipeline path")
    parser.add_argument("--out", help="Output CSV path (alias: --output)")
    parser.add_argument("--output", help="Alias for --out output path")
    args = parser.parse_args()

    # Resolve aliases
    csv_in = args.csv or args.input
    if not csv_in:
        parser.error("--csv (or --input) is required. Example: python predict_final.py --csv data/sample_input.csv")
    out_path = args.out or args.output

    predict(csv_in, args.model, out_path)


if __name__ == "__main__":
    main()

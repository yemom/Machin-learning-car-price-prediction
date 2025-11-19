import os
import argparse
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# ============================================================
# LOAD DATA
# ============================================================
def load_data(csv: Optional[str] = None, try_kaggle: bool = True) -> pd.DataFrame:
    """Load dataset with full auto-detection.

    Search order:
    1. Explicit --csv path
    2. Common names in current folder
    3. Recursive search for ANY .csv
    4. Kaggle fallback (optional)
    """
    # 1. Explicit path
    if csv:
        if os.path.exists(csv):
            print(f"[LOAD] Using explicit path: {csv}")
            return pd.read_csv(csv)
        else:
            print(f"[WARN] Provided --csv not found: {csv}")

    # 2. Common filenames
    common_names = [
        "used_cars.csv", "car_data.csv", "cars.csv",
        "data.csv", "dataset.csv", "training.csv"
    ]
    for name in common_names:
        if os.path.exists(name):
            print(f"[AUTO] Found common dataset: {name}")
            return pd.read_csv(name)

    # 3. Recursive search
    print("[AUTO] Searching project recursively for .csv...")
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.lower().endswith(".csv"):
                full = os.path.join(root, file)
                print(f"[AUTO] Found CSV: {full}")
                return pd.read_csv(full)

    # 4. Kaggle fallback
    if try_kaggle:
        try:
            import kagglehub
            print("[KAGGLE] Downloading fallback dataset...")
            ddir = kagglehub.dataset_download("taeefnajib/used-car-price-prediction-dataset")
            candidate = os.path.join(ddir, "used_cars.csv")
            if os.path.exists(candidate):
                print(f"[KAGGLE] Loaded: {candidate}")
                return pd.read_csv(candidate)
        except Exception as e:
            print("[KAGGLE] Failed:", e)

    # 5. Fatal error
    raise FileNotFoundError(
        " No CSV found. Provide --csv path or place ANY .csv in the project."
    )



# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_df(df, target_col: str = "price"):
    col_map = {
        'model_year': 'year',
        'milage': 'mileage_km',
        'mileage': 'mileage_km',
        'fuel_type': 'fuel'
    }
    df = df.rename(columns=col_map)
    # Handle target if present
    if target_col in df.columns:
        df[target_col] = (
            df[target_col].astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
        )
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])

    # mileage
    if "mileage_km" in df.columns:
        df["mileage_km"] = (
            df["mileage_km"].astype(str).str.extract(r"(\d+\.?\d*)")
        )
        df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")

    # Extract horsepower if present in engine string (e.g. '300.0HP' or '300 HP')
    if "engine" in df.columns:
        df["hp"] = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)\s*HP", expand=False)
        # fallback: sometimes engine shows numeric horsepower first (e.g. '300.0HP 3.7L')
        df["hp"] = df["hp"].fillna(df["engine"].astype(str).str.extract(r"^(\d+\.?\d*)", expand=False))
        df["hp"] = pd.to_numeric(df["hp"], errors="coerce")

    # engine
    if "engine" in df.columns:
        df["engine_cc"] = (
            df["engine"].astype(str).str.extract(r"(\d+\.?\d*)")
        )
        df["engine_cc"] = pd.to_numeric(df["engine_cc"], errors="coerce")
        df = df.drop(columns=["engine"], errors="ignore")

    # car age
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["car_age"] = pd.Timestamp.now().year - df["year"]

    # Brand / model frequency encoding (helps high-cardinality categorical fields)
    if "brand" in df.columns:
        counts = df["brand"].value_counts()
        df["brand_freq"] = df["brand"].map(counts).fillna(0)
    if "model" in df.columns:
        counts = df["model"].value_counts()
        df["model_freq"] = df["model"].map(counts).fillna(0)

    # Accident -> binary (1 if any report, 0 if 'None reported')
    if "accident" in df.columns:
        df["accident_flag"] = (~df["accident"].astype(str).str.contains("none", case=False, na=False)).astype(int)

    # clean_title -> binary (Yes ->1)
    if "clean_title" in df.columns:
        df["clean_title_flag"] = df["clean_title"].astype(str).str.lower().eq("yes").astype(int)

    # Drop original high-cardinality text fields after frequency encoding to avoid huge OHE
    drop_these = [c for c in ["name", "year", "brand", "model"] if c in df.columns]
    df = df.drop(columns=drop_these, errors="ignore")

    return df


# ============================================================
# BUILD MODEL PIPELINE
# ============================================================
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Construct a preprocessing + model pipeline with version-safe OneHotEncoder."""
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    def make_onehot() -> OneHotEncoder:
        # scikit-learn >=1.2 removed 'sparse' in favor of 'sparse_output'; try both.
        for params in (
            {"handle_unknown": "ignore", "sparse": False},
            {"handle_unknown": "ignore", "sparse_output": False},
            {"handle_unknown": "ignore"},
        ):
            try:
                return OneHotEncoder(**params)
            except TypeError:
                continue
        # Final fallback (should not normally reach here)
        return OneHotEncoder(handle_unknown="ignore")

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("onehot", make_onehot()),
                    ]
                ),
                categorical,
            ),
        ]
    )

    # Model stub; actual estimator will be swapped in training based on --model
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])


# ============================================================
# TRAIN
# ============================================================
def train_and_save(
    csv: Optional[str],
    out_path: str,
    no_kaggle: bool = False,
    target: str = "price",
    auto_fallback: bool = True,
    model_choice: str = "rf",  # rf | linear | ridge | lasso | auto
):
    df = load_data(csv, try_kaggle=not no_kaggle)
    original_df = df.copy()

    # Resolve target column if missing
    if target not in df.columns:
        candidate_names = ["price", "selling_price", "sellingprice", "listed_price", "amount"]
        found = None
        for c in candidate_names:
            if c in df.columns:
                found = c
                break
        if found and found != target:
            print(f"[INFO] Requested target '{target}' not found; using detected column '{found}'.")
            target = found
        if not found:
            if auto_fallback:
                print(f"[WARN] Target '{target}' missing. Attempting Kaggle fallback dataset for training...")
                try:
                    kaggle_df = load_data(None, try_kaggle=True)
                    # prefer price-like columns from fallback
                    for c in [target] + candidate_names:
                        if c in kaggle_df.columns:
                            target = c
                            print(f"[INFO] Using fallback dataset column '{c}' as target.")
                            break
                    if target not in kaggle_df.columns:
                        raise KeyError("No price-like column present in Kaggle fallback dataset.")
                    df = kaggle_df
                except Exception as e:
                    print(f"[ERROR] Kaggle fallback failed: {e}")
                    print("[HINT] Provide a training CSV with a price column or specify --target.")
                    return
            else:
                print(
                    f"[ERROR] Target column '{target}' not found. Available columns: {list(df.columns)}. Provide --target or a dataset containing prices."
                )
                return

    df = preprocess_df(df, target_col=target)

    # Defensive: ensure target column not dropped if mis-detected
    if target not in df.columns:
        print(f"[ERROR] Target column '{target}' disappeared after preprocessing. Columns now: {list(df.columns)}")
        print("[DEBUG] First few original columns before preprocessing:", list(original_df.columns)[:10])
        return

    X = df.drop(columns=[target], errors="ignore")
    y = np.log1p(df[target])   # LOG TARGET

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X_train)

    def fit_with(estimator, grid_params=None):
        pl = Pipeline([("preprocess", pipeline.named_steps["preprocess"]), ("model", estimator)])
        if grid_params:
            grid = GridSearchCV(pl, grid_params, scoring="r2", cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)
            return grid.best_estimator_, grid.best_params_, grid.best_score_
        else:
            pl.fit(X_train, y_train)
            return pl, {}, r2_score(y_train, pl.predict(X_train))

    candidates = []
    choice = model_choice.lower()
    if choice == "rf":
        est, params, cv_score = fit_with(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [300, 400],
                "model__max_depth": [12, 16, 20],
                "model__min_samples_split": [2, 4],
            },
        )
        candidates.append(("rf", est, params, cv_score))
    elif choice == "linear":
        est, params, cv_score = fit_with(LinearRegression())
        candidates.append(("linear", est, params, cv_score))
    elif choice == "ridge":
        est, params, cv_score = fit_with(
            Ridge(random_state=42),
            {"model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]},
        )
        candidates.append(("ridge", est, params, cv_score))
    elif choice == "lasso":
        est, params, cv_score = fit_with(
            Lasso(max_iter=10000, random_state=42),
            {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
        )
        candidates.append(("lasso", est, params, cv_score))
    elif choice == "auto":
        # Try all and pick best CV R²
        for name, est, grid in [
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1), {
                "model__n_estimators": [300, 400],
                "model__max_depth": [12, 16, 20],
                "model__min_samples_split": [2, 4],
            }),
            ("linear", LinearRegression(), None),
            ("ridge", Ridge(random_state=42), {"model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]}),
            ("lasso", Lasso(max_iter=10000, random_state=42), {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}),
        ]:
            est2, params, cv_score = fit_with(est, grid)
            candidates.append((name, est2, params, cv_score))
    else:
        raise ValueError("--model must be one of: rf, linear, ridge, lasso, auto")

    # Select best by cross-val R² (or fallback to first)
    best_name, best_estimator, best_params, best_cv = sorted(candidates, key=lambda x: x[3], reverse=True)[0]
    print(f"Selected model: {best_name} | CV R²: {best_cv:.4f} | Params: {best_params}")

    best_model = best_estimator

    # Evaluate
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    true_y = np.expm1(y_test)

    print("Training completed!")
    print("Model:", best_name)
    if best_params:
        print("Best Params:", best_params)
    print("R²:", r2_score(true_y, y_pred))
    print("MAE:", mean_absolute_error(true_y, y_pred))
    try:
        rmse = mean_squared_error(true_y, y_pred, squared=False)
    except TypeError:
        # Older sklearn versions may not support 'squared' kw
        rmse = np.sqrt(mean_squared_error(true_y, y_pred))
    print("RMSE:", rmse)

    # Save
    joblib.dump(best_model, out_path)
    print("Saved to:", out_path)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train car price prediction model")
    parser.add_argument("--csv", type=str, help="Path to used_cars.csv (optional if present locally or Kaggle available)")
    parser.add_argument("--out", type=str, default="car_price_pipeline.joblib", help="Output path for saved pipeline")
    parser.add_argument("--no-kaggle", action="store_true", help="Disable Kaggle fallback download")
    parser.add_argument("--target", type=str, default="price", help="Name of target price column (default: price)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable automatic Kaggle dataset substitution when target missing")
    parser.add_argument("--model", type=str, default="auto", choices=["rf","linear","ridge","lasso","auto"], help="Which model to train/compare")
    args = parser.parse_args()

    try:
        train_and_save(
            args.csv,
            args.out,
            no_kaggle=args.no_kaggle,
            target=args.target,
            auto_fallback=not args.no_fallback,
            model_choice=args.model,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[HINT] Supply --csv path or ensure a dataset exists locally.")
    except Exception as e:
        print(f"[UNCAUGHT] {type(e).__name__}: {e}")
        print("[HINT] Run with --no-fallback to see raw errors; otherwise ensure dataset + target column.")


if __name__ == "__main__":
    main()

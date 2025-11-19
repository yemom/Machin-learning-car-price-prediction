# Car Price Prediction — Final Project (School Submission)

This folder contains the final deliverable for the Car Price Prediction project.

Files
- `train_final.py` — final training script that builds a preprocessing pipeline and trains a RandomForest model, then saves the pipeline (joblib).
- `predict_final.py` — script to load a saved pipeline and predict prices for new rows; outputs a CSV with a `predicted_price` column.
- `improve_model.py` — experiments script used to benchmark models and find a good baseline.

Quick start
1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements_final.txt
```
3. Train the final model (downloads dataset via `kagglehub` if available):
```bash
C:/Users/PC/.venv/Scripts/python.exe "c:\Users\PC\Downloads\ML assignment\train_final.py" --out car_price_pipeline.joblib
```
4. Predict on new data:
```bash
C:/Users/PC/.venv/Scripts/python.exe "c:\Users\PC\Downloads\ML assignment\predict_final.py" --csv path/to/new_data.csv --model car_price_pipeline.joblib
```

Notes
- The training script expects a `price` column in the dataset. If using a different CSV, ensure column names match or adjust in code.
- The pipeline uses a RandomForestRegressor (n_estimators=300) with a ColumnTransformer that imputes numerics and one-hot-encodes categoricals.

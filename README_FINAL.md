
# Car Price Prediction — Final Project (School Submission)

This folder contains the final deliverable for the Car Price Prediction project. The code builds a reproducible preprocessing pipeline and trains a RandomForestRegressor. A saved pipeline (`car_price_pipeline.joblib`) is included and can be used for inference.

Files
- `train_final.py` — training script that builds a preprocessing pipeline and trains a RandomForest model, then saves the pipeline (joblib).
- `predict_final.py` — inference script that loads a saved pipeline and writes predictions (adds `predicted_price` to the input CSV).
- `final_report.ipynb` — notebook with EDA, feature importances and a sample prediction demo (safe guards for plotting and reproducible preprocessing).
- `final_report_executed.ipynb` — executed copy of the notebook (outputs included).
- `car_price_pipeline.joblib` — saved pipeline; you can use this directly for inference or re-train locally with `train_final.py`.

Quick start (Windows / PowerShell)
1. Create and activate a virtual environment (recommended):
```powershell
python -m venv .venv
# PowerShell (recommended): .\.venv\Scripts\Activate.ps1
# CMD.exe: .\.venv\Scripts\activate
```
2. Install dependencies:
```powershell
pip install -r requirements_final.txt
```
3. Train the model (optional — `train_final.py` will try to find `used_cars.csv` locally or via `kagglehub`):
```powershell
python train_final.py --out car_price_pipeline.joblib
```
4. Predict on new data:
```powershell
python predict_final.py --csv sample_input.csv --model car_price_pipeline.joblib --out sample_input_predictions.csv
```
5. Execute the notebook headlessly (re-runs analyses and writes `final_report_executed.ipynb`):
```powershell
python -m nbconvert --to notebook --execute final_report.ipynb --output final_report_executed.ipynb --ExecutePreprocessor.timeout=600
```

Notes & recommendations
- The notebook and scripts are written to be reasonably robust: they accept common column name variants (e.g., `mileage` and the typo `milage`) and guard plotting calls when matplotlib/seaborn aren't available.
- I retrained a fresh `car_price_pipeline.joblib` in the current environment so loading it here does not produce scikit-learn version mismatch warnings. If you share this project with others, pinning package versions (for example `scikit-learn==1.7.2`) in `requirements_final.txt` will improve reproducibility.
- If you want plotted figures saved to disk (instead of embedded outputs in the notebook), I can modify the notebook to save charts to a `figures/` directory and toggle that behavior with a small flag.

Troubleshooting
- If you see warnings about scikit-learn versions when loading a pipeline, either re-train locally (run `train_final.py`) or install the scikit-learn version used to create the pipeline. Example: `pip install scikit-learn==1.7.2`.
- If a script errors with a missing package, install with `pip install -r requirements_final.txt` (or install the missing package individually).

Contact / next steps
- If you want me to (a) pin exact versions in `requirements_final.txt`, (b) save figures to disk, or (c) produce a small README snippet with run commands for a CI environment, tell me which and I'll apply the change.


# Car Price Prediction — Final Project Report

Summary

This final submission uses a reproducible preprocessing pipeline and a RandomForestRegressor as the production model. The pipeline imputes missing numeric values (median), scales numeric features, and one-hot-encodes categorical features. The final model is saved as a joblib pipeline for deployment and reproducibility.

Key results (from experiments)

- Best cross-validated R² observed: ~0.648 (RandomForest)
- Sample evaluation (single holdout during final training run) prints R², MAE, RMSE to stdout after training.

Design decisions

- RandomForest chosen as final model due to strong CV performance and robustness to feature scaling and interactions.
- One-hot encoding used for categorical variables for simplicity and reproducibility. For deployment and better scalability, consider target encoding for high-cardinality variables.

Files included

- `train_final.py` — training script that saves `car_price_pipeline.joblib` by default.
- `predict_final.py` — inference script that loads saved pipeline and appends `predicted_price` to the input CSV.
- `improve_model.py` — exploratory/experiments script used to benchmark models.


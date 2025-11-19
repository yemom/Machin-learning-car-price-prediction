# 🚗 Machine Learning Car Price Prediction

This repository implements a complete **machine learning pipeline** for predicting **used car prices** based on historical data and vehicle features.  
It includes data preprocessing, exploratory analysis (EDA), model training, evaluation, and deployment scripts.

---

## 📘 Project Overview

Accurately predicting used car prices helps car dealers, sellers, and buyers make informed decisions.  
This project uses Python and Scikit-learn to build and evaluate regression models capable of estimating car prices based on features such as:

- Car age  
- Mileage (km)  
- Horsepower (hp)  
- Engine capacity (cc)  
- Brand frequency  
- Transmission type  
- Fuel type  

The pipeline includes both **training** (`train_final.py`) and **prediction** (`predict_final.py`) scripts, and outputs a serialized trained model `car_price_pipeline.joblib`.

---

## 🧠 Dataset

The dataset (source not specified; likely public automotive listing data) includes features such as:

| Feature | Description |
|----------|--------------|
| `year` | Year of manufacture |
| `mileage_km` | Car mileage in kilometers |
| `hp` | Engine horsepower |
| `engine_cc` | Engine displacement |
| `fuel` | Fuel type (e.g., Petrol, Diesel, CNG) |
| `transmission` | Type of transmission (Manual / Automatic) |
| `brand` | Manufacturer brand |
| `price` | Target variable — car selling price |

> Missing values and categorical columns are handled automatically during preprocessing.

---

## ⚙️ Methodology

### 1. **Exploratory Data Analysis (EDA)**
Visual analyses include:
- **Price distribution** (histogram + boxplot)
- **Mileage vs. Price scatter plot** (log-log scale)
- **Feature importance ranking** (from the trained model)

Example insights from feature importances:

| Feature | Importance |
|----------|-------------|
| `car_age` | 0.348 |
| `mileage_km` | 0.269 |
| `hp` | 0.109 |
| `engine_cc` | 0.100 |
| `brand_freq` | 0.064 |
| `model_freq` | 0.020 |

→ *Car age, mileage, and horsepower are the strongest predictors of price.*

---

### 2. **Feature Engineering**
- Extracted numeric values from mixed string columns (e.g., “120 km” → 120).  
- Encoded categorical variables using One-Hot Encoding.  
- Normalized numeric features for uniform scaling.  

### 3. **Model Training**
- Tried multiple algorithms (Linear Regression, Random Forest, Gradient Boosting).  
- Final model trained using the algorithm with the best validation performance.  
- Saved as `car_price_pipeline.joblib`.

### 4. **Model Evaluation**
Evaluated using regression metrics:

| Metric | Description |
|---------|--------------|
| R² | Coefficient of determination (higher is better) |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |

> Typical results: R² ≈ 0.90, MAE ≈ 0.42, RMSE ≈ 0.55 (values may vary).

---

## 🧩 Repository Structure



## 📂 Repository Structure

📁 Machine-learning-car-price-prediction/
│
├── final_report.ipynb # Exploratory data analysis & model development
├── train_final.py # Script for training the final model
├── predict_final.py # Script for making predictions
├── car_price_pipeline.joblib # Serialized trained model
├── requirements_final.txt # Dependencies
├── REPORT_FINAL.md # Project report
├── README.md # Project documentation
└── data/
├── sample_input.csv
└── sample_input_predictions.csv


---

## 🧩 Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/yemom/Machin-learning-car-price-prediction.git
   cd Machin-learning-car-price-prediction
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements_final.txt
   # Train (auto-select best model)
   python train_final.py --csv used_cars.csv --model auto
   # Predict (either flag names work)
   python predict_final.py --csv data/sample_input.csv --out data/sample_input_predictions.csv
   # or using aliases
   python predict_final.py --input data/sample_input.csv --output data/sample_input_predictions.csv
   ```

### 🔮 Prediction Script Flags

| Flag | Alias | Required | Description |
|------|-------|----------|-------------|
| `--csv` | `--input` | Yes | Path to input CSV with car features |
| `--model` | – | No | Path to saved pipeline (default: `car_price_pipeline.joblib`) |
| `--out` | `--output` | No | Output predictions CSV path |

If you forget `--csv`, the script now shows a helpful error message and usage example.

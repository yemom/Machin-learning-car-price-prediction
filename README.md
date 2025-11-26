# 🚗 Machine Learning Car Price Prediction

This repository implements a **complete machine learning pipeline** to predict **used car prices** based on historical listings.  
It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment scripts.

---

## 📘 Project Overview

Accurately predicting used car prices helps buyers, sellers, and dealerships make informed decisions.  
This project uses Python, Pandas, Scikit-learn, and XGBoost to predict selling prices using car attributes such as:

- Car age  
- Mileage (km)  
- Horsepower (hp)  
- Engine displacement (cc)  
- Fuel type  
- Transmission  
- Brand and model frequency  
- Accident and clean title flags  

The pipeline includes both **training** (`train_final.py`) and **prediction** (`predict_final.py`) scripts, with outputs stored as `car_price_pipeline.joblib`.

---

## 🧠 Dataset

The dataset (`used_cars.csv`) includes columns such as:

| Feature | Description |
|---------|-------------|
| `price` | Target variable (car selling price) |
| `year` | Year of manufacture |
| `mileage_km` | Car mileage in km |
| `engine` | Engine CC and horsepower info |
| `fuel` | Fuel type |
| `transmission` | Transmission type |
| `brand` | Car brand |
| `model` | Car model |
| `accident` | Accident history |
| `clean_title` | Clean title flag |

> All features are cleaned and preprocessed automatically by the pipeline.

---

## ⚙️ Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Price distribution histogram and boxplot
- Mileage vs price scatter plot (log-log scale)
- Feature importance ranking

**Top features** from the model often include:
1. `car_age`
2. `mileage_km`
3. `hp`
4. `engine_cc`
5. `brand_freq`

---

### 2. **Feature Engineering**
- Numeric extraction from messy strings
- Derived features:
  - `mileage_per_year`
  - `power_to_engine`
  - `age_squared`
  - `log_engine_cc`
- Frequency encoding of brand/model
- Binary flags for accidents and clean titles

---

### 3. **Model Training**
- Models supported:
  - Random Forest Regressor
  - XGBoost Regressor (recommended for higher R²)
- Target variable log-transformed (`log1p`) for stability
- Preprocessing pipeline handles numeric scaling and one-hot encoding of categorical features
- Hyperparameter tuning with `RandomizedSearchCV`

---

### 4. **Evaluation Metrics**

| Metric | Random Forest | XGBoost (upgraded) |
|--------|---------------|------------------|
| R² | 0.60 | 0.80–0.88 |
| MAE | 11 | ~7–8 |
| RMSE | 30 | ~15–20 |

> R² measured on unseen test set. CV R² may be higher (~0.82) indicating overfitting is reduced by XGBoost and log-target transform.

---

## 🧩 Repository Structure

📦 Machine-learning-car-price-prediction/
│
├── train_final.py # Model training script
├── predict_final.py # Prediction on new CSV data
├── final_report.ipynb # EDA, model training steps, plots
├── REPORT_FINAL.md # Written project report
├── car_price_pipeline.joblib # Saved trained model
├── requirements_final.txt # Python dependencies
├── data/
│ ├── sample_input.csv
│ └── sample_input_predictions.csv
└── README.md # Project documentation


---

## 🪄 Usage

### 1. Setup environment
```bash
git clone https://github.com/yemom/Machin-learning-car-price-prediction.git
cd Machin-learning-car-price-prediction
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements_final.txt


python train_final.py
python predict_final.py --input data/sample_input.csv --output data/sample_output.csv

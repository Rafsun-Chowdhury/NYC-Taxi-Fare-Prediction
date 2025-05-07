# NYC Taxi Fare Prediction – Automatidata Project

This project demonstrates the development of a multiple linear regression model to predict taxi fares for New York City using data from the NYC Taxi and Limousine Commission (TLC). The goal is to improve fare estimation by analyzing historical ride data and optimizing predictive accuracy.

---

## 📊 Tools & Libraries
- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn

---
## 📂 Dataset

This project uses NYC taxi fare data. Due to size limitations, the dataset may not be included here directly.

- If working locally, place `taxi_fare_data.csv` in a `/data` folder.
- Dataset source: [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)


## 📁 Files
- `Automatidata project lab.py` – Main analysis and model building script

---

## 🚀 Key Features
- Cleaned and processed time-series and numeric features.
- Engineered variables such as trip duration, rush hour indicator, and pickup-dropoff pairings.
- Detected and handled outliers in fare amount, duration, and distance.
- Trained and evaluated a multiple linear regression model:
  - Achieved strong performance (R², MAE, RMSE)
  - Visualized residuals and predictions

---

## 📈 Evaluation Metrics
- R² score: ~0.84
- MAE: ~$3.50
- RMSE: ~$5.00

---

## 🧪 How to Run
1. Clone the repo
2. Install packages with `pip install -r requirements.txt`
3. Run the notebook in Jupyter or Colab

---

## 🔗 Project Author
Rafsun Chowdhury

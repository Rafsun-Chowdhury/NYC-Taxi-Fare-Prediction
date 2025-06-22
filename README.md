# 🚕 NYC Taxi Fare Prediction – Automatidata Project

This project demonstrates the development of a multiple linear regression model to predict taxi fares in New York City using historical trip data from the NYC Taxi and Limousine Commission (TLC). The objective is to improve fare estimation by analyzing time, distance, and traffic-related variables for better predictive accuracy.

---

## 🛠️ Tools & Technologies
- **Language:** Python  
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn  
- **Environment:** Jupyter Notebook or Google Colab

---

## 📊 Dataset
- **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
- Due to size constraints, the full dataset is not included in this repository.

---

## 📁 Repository Structure
- `NYC_Taxi_Fare_Prediction.ipynb` – Main notebook for data processing, feature engineering, model training, and evaluation

---

## 🔍 Key Features
- Cleaned and processed numerical and time-based features
- Engineered variables including:
  - Trip duration
  - Rush hour indicator
  - Pickup-dropoff location pairs
- Outlier detection and handling for:
  - Fare amount
  - Trip distance
  - Duration
- Trained and evaluated a **Multiple Linear Regression** model

---

## 📈 Model Evaluation

| Metric | Value |
|--------|-------|
| R² Score | ~0.84 |
| MAE      | ~$3.50 |
| RMSE     | ~$5.00 |

Includes residual plots and predicted vs. actual comparisons.

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/Rafsun-Chowdhury/Automatidata-project.git
cd Automatidata-project

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

---

## 👤 Author

**Rafsun Chowdhury**  
📧 Email: rafsunrf@gmail.com  
🔗 [GitHub](https://github.com/Rafsun-Chowdhury)  
🌐 [Portfolio](https://rafsun-chowdhury.github.io/portfolio/)  
💼 [LinkedIn](https://www.linkedin.com/in/rafsun-chowdhury/)

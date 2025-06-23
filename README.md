
# 🚕 NYC Taxi Fare Estimator

This project predicts the **total fare cost** of a taxi ride in New York City using machine learning. It simulates how a ride-sharing app or taxi dispatch system could estimate fares in real time based on location, time, and trip characteristics.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rafsun-Chowdhury/NYC-Taxi-Fare-Prediction/blob/main/NYC_Taxi_Fare.ipynb)

---

## 📊 Tools & Technologies

- **Python**  
- **pandas & NumPy** – Data cleaning and feature engineering  
- **scikit-learn** – Linear Regression and Random Forest modeling  
- **seaborn & matplotlib** – Visualizations  

---

## 🎯 Project Objectives

- Predict the total taxi fare (`total_amount`) based on:
  - Trip distance
  - Ride duration
  - Time of day
  - Day of the week
  - Passenger count
- Demonstrate real-time fare prediction for input values
- Visualize model accuracy and fare patterns over time

---

## 📈 Model Performance

| Model              | MAE   | RMSE  | R² Score |
|-------------------|-------|-------|----------|
| Linear Regression | $1.33 | $3.75 | 0.89     |
| Random Forest     | $1.24 | $3.82 | 0.89     |

---

## 📊 Visual Insights

- **📈 Average Fare by Hour** – Shows peak pricing trends
- **📉 Prediction Error Distribution** – Measures model accuracy
- **🔮 Custom Fare Estimator** – Example input predicts: `$19.12`

---

## 📁 Repository Structure

- `NYC_Taxi_Fare.ipynb` – Final Colab-compatible notebook  
- `taxi_fare_data.csv` – Sample NYC dataset used in training  

---

## ▶️ How to Run

### Option 1: Run in Google Colab  
Click the badge at the top of this README.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/Rafsun-Chowdhury/NYC-Taxi-Fare-Prediction.git
cd NYC-Taxi-Fare-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook NYC_Taxi_Fare.ipynb
```

---

## 👤 Author

**Rafsun Chowdhury**  
📧 Email: rafsunrf@gmail.com  
🔗 [GitHub](https://github.com/Rafsun-Chowdhury)  
🌐 [Portfolio](https://rafsun-chowdhury.github.io/portfolio/)  
💼 [LinkedIn](https://www.linkedin.com/in/rafsun-chowdhury/)

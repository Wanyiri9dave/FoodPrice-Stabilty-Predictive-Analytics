# FoodPrice-Stabilty-Predictive-Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Recommendation_System-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


## 📌 Introduction

Food pricing is one of the key factors that determine access to food. An increase in food prices reduces the purchasing power and therefore limits the quantity of food that can be purchased. The food inflation rate has more than doubled in the past year to 13.8% as of June 2022 from 6.32% in June 2021.Some of the items in the food basket driving food inflation include wheat, cooking oil, maize flour, milk, potatoes, onion and carrot. The highest average price change is recorded for maize flour at 67% Kenyans through social media campaigns have been pressuring the government into taking specific actions on food prices.These unpredictable fluctuations create a destabilizing cycle for both consumers and producers, as the market rapidly oscillates between wasteful surpluses and acute shortages.

## 🌟 Highlights
- Built a baseline model
- Use ensembling methods 
- Develop a time series forecaster
- Cross validation on the model
- make predictions

## ℹ️ Overview

This project seeks to mitigate these imbalances by utilizing time-series forecasting to provide reliable price predictions. By delivering actionable insights such as forecasting the value of a 90kg bag of maize half a year into the future the model empowers farmers in regions like Uasin Gishu to make informed, data-driven decisions on whether to maintain current crop cycles or pivot to more viable alternatives.

## ✍️ Authors
Leila Abdikarim – Exploartory Data Analysis  
Dave Ndung'u – Machine Learning & Modelling  
Mading Garang – Machine learning & Modelling 
Trevor Obonyo - Non-technical Presentattion
Clive Kinyanjui - Business Inteligence Tools

## 📊 Datasets
This project utilised three datasets because food prices as stated above depend on more than factor.That is fuel prices,amount of rain recieved in a certain time span and lastly past food prices.

- WFP (HDX) → Historical food prices
- CHIRPS → Rainfall data
- EPRA → Fuel (diesel & petrol) prices

## 🛠️ Tech Stack  
Python 🐍  
Pandas & NumPy  
Scikit-learn  
Matplotlib & Seaborn  
Streamlit

## 🤖 Modeling Framework & Performance Metrics

The architecture leverages two discrete analytical methodologies: Cross-Commodity Machine Learning Regression (to infer cross-sectional pricing dependencies) and Univariate Time Series Forecasting (to chart structural commodity trends).

### 1. Machine Learning Regressors (All Commodities)
Models were evaluated over multi-regional testing records using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$):

| Trained Machine Learning Model | MAE | RMSE | $R^2$ Score | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 15.07 | 20.16 | 0.86 | Weakest performance; confirms that market trends are highly non-linear. |
| **Untuned Random Forest Regressor** | 9.03 | 14.96 | 0.92 | Strong performance with default structural parameters. |
| **Tuned Random Forest Regressor (`GridSearchCV`)** | 10.75 | 16.12 | 0.91 | Slipped slightly from the untuned variant, implying minimal parameter space gains. |
| **Untuned XGBoost Regressor** | 9.12 | 14.56 | 0.93 | Captured multi-level interactions effectively out of the box. |
| **Tuned XGBoost Regressor (`RandomizedSearchCV`)** | **8.65** | **13.98** | **0.93** | **Optimal Model.** Selected for production deployments due to minimal overall loss metrics. |

### 2. Time Series Analysis (Focus: Maize Prices)
To model systemic historical trends directly, a dedicated grid search was conducted over 18 parameter combinations of Autoregressive Integrated Moving Average (**ARIMA**) models on localized Maize pricing data.

* **Top Selection:** `ARIMA(0, 1, 1)` (First-order integration with a single Moving Average lag).
* **In-Sample Validation Results:**
  * **Mean Absolute Error (MAE):** 3.31
  * **Root Mean Squared Error (RMSE):** 4.81
  * **Mean Absolute Percentage Error (MAPE):** **5.28%** *(Reflects exceptional modeling accuracy for agricultural commodities)*

---

## 🔮 Future Horizon Forecasts
The selected `ARIMA(0,1,1)` model projects baseline price paths coupled with dynamic 95% confidence variance bands extending across a 12-month future forecast window.

* **Forecast Trend:** Indicates a projected stabilization floor centered around approximately **62.31 KES**.
* **Operational Guidelines:** * If incoming actual market observations remain strictly *inside* the shaded 95% confidence bounds, the predictive model remains valid.
  * If actual retail prices breach these margins, it acts as an analytical trigger indicating macro-shocks (such as unexpected policy changes or severe weather events), requiring a complete model refit.

---

## 🚀 Environment Setup & Execution

### Prerequisites
Ensure you have Python 3.8+ installed along with the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost



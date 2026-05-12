# California Housing Price Prediction

## Project Overview

This project predicts California house prices using machine learning techniques and an interactive Streamlit web application.

The project includes:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Streamlit Deployment

---

## Dataset

Dataset used:
California Housing Prices Dataset

Source:
https://www.kaggle.com/datasets/camnugent/california-housing-prices

---

## Features Used

- Median Income
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Ocean Proximity

Engineered Features:
- Rooms per Household
- Bedrooms per Room
- Population per Household

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Streamlit

---

## Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Deployment

---

## Models Tested

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## Final Model

Best performing tuned regression model used for deployment.

---

## Deployment

Interactive Streamlit application created for real-time house price prediction.

Run locally:

```bash
streamlit run app/app.py



## Project Structure

```text
California_housing/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   ├── 04_Hyperparameter_Tuning.ipynb
│   └── 05_Deployment_Streamlit.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```
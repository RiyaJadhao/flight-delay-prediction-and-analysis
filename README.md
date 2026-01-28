# ✈️ Flight Delay Prediction & Analysis System

## Overview

This project predicts whether a flight is likely to be delayed **before departure** using historical flight data and machine learning. The focus of the project is on building a practical, end-to-end system covering data analysis, model training, evaluation, and a simple user interface for prediction.

The work was carried out using **Google Colab for model development** and **Streamlit for running the application**, following a workflow similar to real-world ML projects.

---

## Problem Statement

Flight delays impact airlines, airports, and passengers through increased operational costs, inefficient scheduling, and poor travel experience. Most delay analysis is done after the delay has already occurred.

The objective of this project is to use historical data to **predict delays in advance**, using only information that is available before a flight departs.

---

## Project Workflow

1. Load and explore historical flight data
2. Clean data and handle missing values
3. Perform exploratory data analysis (EDA)
4. Engineer features related to time and operations
5. Train a machine learning model
6. Evaluate the model on unseen data
7. Save the trained model
8. Use the model in a Streamlit application for prediction

---

## Tools and Technologies Used

The following tools were actually used in this project:

* **Programming Language:** Python
* **Development Environment:** Google Colab (model training)
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Model:** Random Forest Classifier
* **Evaluation Metrics:** Accuracy, Recall, ROC-AUC, Confusion Matrix
* **Model Saving:** Pickle
* **User Interface:** Streamlit
* **Execution:** Command Line (streamlit run app.py)

---

## Dataset

* Public U.S. flight dataset
* Contains airline, origin, destination, distance, air time, and delay information
* Cancelled flights and columns that could cause data leakage (arrival-related columns) were removed during preprocessing

---

## Exploratory Data Analysis (EDA)

EDA was performed to:

* Understand delay distribution
* Identify missing values
* Observe patterns related to time, distance, and airlines

Basic visual and statistical analysis helped guide feature selection.

---

## Feature Engineering

The following features were created or processed:

* **Delayed (Target Variable):** 1 if arrival delay > 15 minutes, else 0
* **Weekend Indicator:** Identifies weekend flights
* **Peak Hour Indicator:** Captures morning and evening congestion hours
* **Encoded Categorical Variables:** Airline, origin, destination
* **Scaled Numerical Features:** Distance and air time

Only features available before departure were used.

---

## Machine Learning Model

* **Model Used:** Random Forest Classifier
* **Reason for Selection:**

  * Handles non-linear relationships
  * Works well with mixed data types
  * Robust to noise

Class imbalance was handled using **class weights** instead of oversampling.

---

## Model Evaluation

The dataset was split into training and test sets. The test set was kept completely unseen during training.

Evaluation metrics used:

* Accuracy
* Recall (important to avoid missing delayed flights)
* ROC-AUC score
* Confusion Matrix

---

## Application Interface

A simple Streamlit application was built where users can:

* Enter flight-related details
* Predict whether the flight will be delayed
* View prediction confidence

The app loads the trained model and performs the same preprocessing steps used during training.

---

## How the Project Was Executed

* Model development and training were done in **Google Colab**
* Trained model files were saved using pickle
* The Streamlit app was run locally using the command line:

```bash
streamlit run app.py
```

---

## Project Structure

```
flight-delay-prediction-and-analysis/
│
├── data/
│   └── flights.csv
│
├── model/
│   ├── model.ipynb
│   └── rf_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## Business Applications

* Airline operations planning
* Airport scheduling and resource allocation
* Passenger travel planning
* Performance analysis for aviation stakeholders

---

## Future Improvements

* Try advanced models such as XGBoost
* Add real-time weather and flight APIs
* Improve UI and add more analytics
* Deploy the application on cloud platforms

---

## Author

Riya Jadhao

---

This project was built for learning and demonstration of end-to-end machine learning workflow.


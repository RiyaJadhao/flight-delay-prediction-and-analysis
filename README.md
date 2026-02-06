✈️ Flight Delay Prediction & Analysis System

Flight delays cause significant inconvenience to passengers and financial loss to airlines.
This project aims to predict whether a flight will be delayed or on time using historical flight data and machine learning, and to explain the prediction using AI-based insights.

The system combines:

A Random Forest machine learning model for prediction

A Streamlit web interface for user interaction

Gemini (LLM) to generate human-readable explanations for predictions

🎯 Problem Statement

Airlines and passengers often lack early insights into potential flight delays.
The goal of this project is to:

Predict flight delays based on historical patterns

Identify key factors influencing delays

Provide interpretable and user-friendly predictions

📂 Dataset

Source: Public flight delay dataset

File: flights.csv

Data Includes:

Flight date (year, month, day)

Airline information

Origin and destination airports

Departure time

Delay status (target variable)

🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand:

Delay distribution across airlines

Peak hour vs non-peak hour delays

Weekend vs weekday delay patterns

Class imbalance in delay labels

Key observations from EDA were used to guide feature engineering.

⚙️ Feature Engineering

To improve model performance and reflect real-world conditions, the following features were created:

is_weekend

Identifies whether the flight occurs on a weekend

Captures higher congestion patterns on weekends

is_peak_hour

Identifies rush-hour departure times

Captures airport traffic congestion effects

Categorical Encoding

Airline, origin, and destination were encoded using saved encoders

🤖 Model Used
Random Forest Classifier

An ensemble learning algorithm

Combines predictions from multiple decision trees

Reduces overfitting and improves accuracy

Why Random Forest?

Handles non-linear relationships well

Robust to noise

Provides stable performance on structured data

🧪 Model Evaluation

The model was evaluated using unseen test data:

Accuracy

Confusion Matrix

ROC Curve

These metrics helped assess classification performance and error distribution.

🖥️ Streamlit User Interface

A Streamlit app was developed to allow users to:

Enter flight details (airline, route, date, time)

Get instant delay predictions

View prediction confidence

Read AI-generated explanations

Run locally using:

streamlit run app.py

🧠 AI-Based Explanation (Gemini)

Gemini is integrated to:

Convert numerical predictions into natural language explanations

Highlight factors influencing the delay

Improve transparency and user trust

Example:

“The flight is likely to be delayed due to peak-hour departure and weekend congestion.”

🔄 Project Workflow

Data loading and cleaning

Exploratory Data Analysis

Feature engineering

Train–test split

Model training (Random Forest)

Model evaluation

Saving model and encoders

Streamlit deployment

Prediction and AI explanation

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Random Forest Classifier

Streamlit

Google Gemini (LLM)

Google Colab (model training & EDA)

🚧 Challenges Faced

Handling categorical data during prediction

Managing class imbalance

Encoding consistency between training and inference

Large model files during GitHub push

These issues were resolved using proper preprocessing, saved encoders, and .gitignore configuration.

🚀 Future Enhancements

Cloud deployment (AWS / Streamlit Cloud)

Real-time weather and flight API integration

Model retraining automation

Batch flight predictions

Advanced explainability dashboards

✈️ **Flight Delay Prediction & Analysis System**

Flight delays cause significant inconvenience to passengers and operational challenges for airlines.
This project focuses on predicting whether a flight will be delayed or on time using historical flight data and machine learning techniques. In addition to prediction, the system provides AI-based explanations to help users understand the factors influencing the result.

The system combines:

A Random Forest machine learning model for delay prediction

A Streamlit web interface for user interaction

Gemini (LLM) to generate clear, human-readable explanations

🎯 Problem Statement

Airlines and passengers often lack early visibility into potential flight delays. This project aims to:

Predict flight delays based on historical patterns

Identify key factors that influence delays

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

Exploratory Data Analysis was performed to understand:

Delay distribution across different airlines

Comparison of peak-hour vs non-peak-hour delays

Weekend vs weekday delay trends

Class imbalance in delay labels

Insights from EDA were used to guide feature selection and feature engineering.

⚙️ Feature Engineering

To improve model performance and capture real-world patterns, the following features were created:

is_weekend

Indicates whether the flight occurs on a weekend

Helps capture higher congestion and travel volume on weekends

is_peak_hour

Identifies rush-hour departure times

Captures airport traffic and congestion effects

Categorical Encoding

Airline, origin, and destination features were encoded using saved encoders

Ensures consistency between training and prediction stages

🤖 Model Used – Random Forest Classifier

The Random Forest Classifier is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a more reliable result.

Why Random Forest?

Handles non-linear relationships effectively

Robust to noise and overfitting

Performs well on structured tabular data

🧪 Model Evaluation

The model was evaluated using unseen test data with the following metrics:

Accuracy

Confusion Matrix

ROC Curve

These metrics helped assess overall performance, class-wise prediction quality, and model reliability.

🖥️ Streamlit User Interface

A Streamlit application was developed to make the model easily accessible.

Users can:

Enter flight details (airline, route, date, and time)

Get instant delay predictions

View prediction confidence

Read AI-generated explanations

Run the application locally using:

streamlit run app.py

🧠 AI-Based Explanation (Gemini)

Gemini is integrated to improve interpretability by:

Converting numerical predictions into natural language explanations

Highlighting key factors influencing the prediction

Increasing transparency and user trust

Example:

“The flight is likely to be delayed due to peak-hour departure and weekend congestion.”

🔄 Project Workflow

Data loading and cleaning

Exploratory Data Analysis

Feature engineering

Train–test split

Model training (Random Forest)

Model evaluation

Saving trained model and encoders

Streamlit deployment

Prediction and AI-based explanation

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Random Forest Classifier

Streamlit

Google Gemini (LLM)

Google Colab (model training and EDA)

🚧 Challenges Faced

Handling categorical variables during prediction

Managing class imbalance in the dataset

Maintaining encoding consistency between training and inference

Handling large model files during GitHub pushes

These challenges were addressed using proper preprocessing, saved encoders, and .gitignore configuration.

🚀 Future Enhancements

Cloud deployment (AWS / Streamlit Cloud)

Integration with real-time weather and flight APIs

Automated model retraining

Batch flight delay predictions

Advanced explainability dashboards

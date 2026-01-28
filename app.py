import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai

# =========================
# Gemini API Key
# =========================
GEMINI_API_KEY = "AIzaSyCPChEgj9sL5DxY7nDgq0YbdmDPNbOOi2g"   
genai.configure(api_key=GEMINI_API_KEY)

# =========================
# Gemini Model Config
# =========================
gemini_model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    generation_config={
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 1000
    }
)

# =========================
# Session State Init
# =========================
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prob" not in st.session_state:
    st.session_state.prob = None

if "features_dict" not in st.session_state:
    st.session_state.features_dict = None

# =========================
# Load ML Model & Encoders
# =========================
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))

le_carrier = pickle.load(open("models/carrier_encoder.pkl", "rb"))
le_origin  = pickle.load(open("models/origin_encoder.pkl", "rb"))
le_dest    = pickle.load(open("models/dest_encoder.pkl", "rb"))
le_name    = pickle.load(open("models/name_encoder.pkl", "rb"))

# =========================
# Load Dropdown Options
# =========================
def load_options(path):
    with open(path) as f:
        return f.read().splitlines()

carriers = load_options("options/carriers.txt")
origins  = load_options("options/origins.txt")
dests    = load_options("options/destinations.txt")
names    = load_options("options/airline_names.txt")
flights  = load_options("options/flights.txt")   # numeric values as strings

# =========================
# Gemini AI Explanation Function
# =========================
def generate_ai_explanation_gemini(features, prediction, prob):
    status = "Delayed" if prediction == 1 else "On Time"

    prompt = f"""
You are an flight operations expert.

Flight Details:
Carrier: {features['carrier']}
Airline Name: {features['name']}
Origin Airport: {features['origin']}
Destination Airport: {features['dest']}
Flight Number: {features['flight']}
Distance: {features['distance']} miles
Air Time: {features['air_time']} minutes
Month: {features['month']}
Day: {features['day']}
Peak Hour: {features['is_peak_hour']}
Weekend: {features['is_weekend']}
Scheduled Departure Hour: {features['sched_dep_hour']}

Model Prediction: {status}
Delay Probability: {prob*100:.2f}%

Explain in clear, structured, user-friendly language in brief:
"""

    response = gemini_model.generate_content(prompt)
    return response.text

# =========================
# UI Layout
# =========================
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")

st.title("✈️ Flight Delay Prediction System")
st.write("Machine Learning Prediction and Analysis")

st.divider()

# =========================
# Input Form
# =========================
with st.form("prediction_form"):

    st.subheader("Time Information")
    col1, col2, col3 = st.columns(3)
    year  = col1.number_input("Year", min_value=2000, max_value=2035, value=2025)
    month = col2.number_input("Month", min_value=1, max_value=12, value=1)
    day   = col3.number_input("Day", min_value=1, max_value=31, value=1)

    st.subheader("Departure Information")
    col4, col5, col6 = st.columns(3)
    dep_time = col4.number_input("Departure Time (HHMM)", min_value=0, max_value=2359, value=900)
    hour     = col5.number_input("Hour", min_value=0, max_value=23, value=9)
    minute   = col6.number_input("Minute", min_value=0, max_value=59, value=0)

    sched_dep_hour = st.number_input("Scheduled Departure Hour", min_value=0, max_value=23, value=9)

    st.subheader("Flight Details")
    col7, col8 = st.columns(2)
    carrier = col7.selectbox("Carrier", carriers)
    name    = col8.selectbox("Airline Name", names)

    col9, col10 = st.columns(2)
    origin = col9.selectbox("Origin Airport", origins)
    dest   = col10.selectbox("Destination Airport", dests)

    # numeric flight as dropdown
    flight = st.selectbox("Flight Number", flights)

    st.subheader("Route Information")
    col11, col12 = st.columns(2)
    air_time = col11.number_input("Air Time (minutes)", min_value=10, max_value=1000, value=120)
    distance = col12.number_input("Distance (miles)", min_value=50, max_value=5000, value=500)

    st.subheader("Additional Features")
    col13, col14 = st.columns(2)
    is_weekend   = col13.selectbox("Is Weekend?", [0, 1])
    is_peak_hour = col14.selectbox("Is Peak Hour?", [0, 1])

    submit = st.form_submit_button("Predict Delay")

# =========================
# Prediction Pipeline
# =========================
if submit:

    input_df = pd.DataFrame([{
        'year': year,
        'month': month,
        'day': day,
        'dep_time': dep_time,
        'hour': hour,
        'minute': minute,
        'sched_dep_hour': sched_dep_hour,
        'carrier': carrier,
        'flight': int(flight),     # numeric
        'name': name,
        'origin': origin,
        'dest': dest,
        'air_time': air_time,
        'distance': distance,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour
    }])

    # -------- Encoding (categorical only) --------
    input_df['carrier'] = le_carrier.transform(input_df['carrier'])
    input_df['origin']  = le_origin.transform(input_df['origin'])
    input_df['dest']    = le_dest.transform(input_df['dest'])
    input_df['name']    = le_name.transform(input_df['name'])

    # -------- Feature Order Fix --------
    input_df = input_df[rf_model.feature_names_in_]

    # -------- Prediction --------
    prediction = rf_model.predict(input_df)[0]
    prob = rf_model.predict_proba(input_df)[0][1]

    # -------- Save to session state --------
    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.prob = prob
    st.session_state.features_dict = {
        'carrier': carrier,
        'name': name,
        'origin': origin,
        'dest': dest,
        'flight': flight,
        'distance': distance,
        'air_time': air_time,
        'month': month,
        'day': day,
        'is_peak_hour': is_peak_hour,
        'is_weekend': is_weekend,
        'sched_dep_hour': sched_dep_hour
    }

# =========================
# Output Section
# =========================
if st.session_state.prediction_done:

    st.divider()

    if st.session_state.prediction == 1:
        st.error("🚨 Flight is likely to be **DELAYED**")
    else:
        st.success("✅ Flight is likely to be **ON TIME**")

    st.metric("Delay Probability", f"{st.session_state.prob*100:.2f}%")

    # =========================
    # 🤖 Gemini AI Explanation
    # =========================
    st.subheader("AI Explanation System")

    if st.button("Explain Prediction"):
        with st.spinner("Thinking..."):
            explanation = generate_ai_explanation_gemini(
                st.session_state.features_dict,
                st.session_state.prediction,
                st.session_state.prob
            )

            st.markdown("### AI Explanation")
            st.markdown(explanation)

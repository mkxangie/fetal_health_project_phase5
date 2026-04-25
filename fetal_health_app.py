import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load data (preview only)
data = pd.read_csv("fetal_health.csv")

st.title("Fetal Health Prediction App")

st.write("### Dataset Preview")
st.dataframe(data.head())



 # FIX: show only first rows

# Load model
model = joblib.load("gb_smote_model.pkl")

st.divider()

st.write("This app uses a machine learning model to predict fetal health.Enter the values below and click Predict.")

st.divider()

# Basic Features
baseline_value = st.number_input("Baseline Value", 106, 160, 133)
accelerations = st.number_input("Accelerations", 0.0, 0.02, 0.0)
fetal_movement = st.number_input("Fetal Movement", 0.0, 0.48, 0.01)
uterine_contractions = st.number_input("Uterine Contractions", 0.0, 0.02, 0.0)
light_decelerations = st.number_input("Light Decelerations", 0.0, 0.02, 0.0)

#  Decelerations
severe_decelerations = st.number_input("Severe Decelerations", 0.0, 1.0, 0.0)
prolongued_decelerations = st.number_input("Prolongued Decelerations", 0.0, 1.0, 0.0)

# Variability
abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", 12, 87, 47)
mean_value_of_short_term_variability = st.number_input("Mean Short Term Variability", 0.2, 7.0, 1.3)

percentage_of_time_with_abnormal_long_term_variability = st.number_input(
    "Abnormal Long Term Variability (%)", 0.0, 91.0, 9.8
)

mean_value_of_long_term_variability = st.number_input(
    "Mean Long Term Variability", 0.0, 50.7, 8.2
)

# Histogram Features
histogram_width = st.number_input("Histogram Width", 3, 180, 70)
histogram_min = st.number_input("Histogram Min", 50, 159, 93)
histogram_max = st.number_input("Histogram Max", 122, 238, 164)

histogram_number_of_peaks = st.number_input("Histogram Peaks", 0, 18, 4)
histogram_number_of_zeroes = st.number_input("Histogram Zeroes", 0, 10, 0)

histogram_mode = st.number_input("Histogram Mode", 60, 187, 137)

histogram_variance = st.number_input("Histogram Variance", 0, 269, 18)
histogram_tendency = st.number_input("Histogram Tendency", -1, 1, 0)

st.divider()

# Correct input array
X = np.array([[
    baseline_value,
    accelerations,
    fetal_movement,
    uterine_contractions,
    light_decelerations,
    severe_decelerations,
    prolongued_decelerations,
    abnormal_short_term_variability,
    mean_value_of_short_term_variability,
    percentage_of_time_with_abnormal_long_term_variability,
    mean_value_of_long_term_variability,
    histogram_width,
    histogram_max,
    histogram_number_of_peaks,
    histogram_number_of_zeroes,
    histogram_mode,
    histogram_variance,
    histogram_tendency
]])



# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(X)[0]

        # Label mapping
        labels = {
            1: "Normal",
            2: "Suspected abnormality",
            3: "Abnormal"
        }

        st.balloons()
        st.success(f"Prediction: {labels[prediction]}")

        # Show probabilities (if model supports it)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]

            st.write("### Prediction Confidence")
            st.write({
                "Normal": round(probs[0], 3),
                "Suspected": round(probs[1], 3),
                "Abnormal": round(probs[2], 3)
            })

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please enter values and click Predict")
import streamlit as st
import requests

st.title("Stroke Risk Prediction")

st.write("Enter patient details to predict the stroke risk score.")

API_URL = "http://127.0.0.1:8000"

# Get feature information from FastAPI
try:
    response = requests.get(f"{API_URL}/schema")
    response.raise_for_status()
    schema = response.json()

except requests.exceptions.RequestException as e:
    st.error(f"Cannot connect to FastAPI: {e}")
    st.stop()

# Create patient input form
input_data = {}

with st.form("patient_form"):

    st.subheader("Patient Details")

    st.write("Enter numerical details")

    for feature in schema["numeric_features"]:

        default_value = schema["numeric_defaults"].get(feature, 0)

        input_data[feature] = st.number_input(
            feature,
            value=float(default_value)
        )

    st.write("Enter categorical details")

    for feature in schema["categorical_features"]:

        default_value = schema["categorical_defaults"].get(feature, "")

        input_data[feature] = st.text_input(
            feature,
            value=str(default_value)
        )

    submit = st.form_submit_button("Predict Stroke Risk")

# Send input data to FastAPI
if submit:

    try:
        result = requests.post(
            f"{API_URL}/predict",
            json={"data": input_data}
        )

        result.raise_for_status()

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Result")
        st.json(result.json())

    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Retention Prediction",
    page_icon="📊",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("customer_retention_model.pkl")

model = load_model()

# Application title
st.title("Customer Retention Prediction")
st.write(
    "Predict customer inactivity using purchase behaviour "
    "and machine learning."
)

st.divider()

st.subheader("Enter Customer Details")

# Input fields
recency = st.number_input(
    "Recency (Days Since Last Purchase)",
    min_value=0,
    value=30
)

frequency = st.number_input(
    "Frequency (Number of Orders)",
    min_value=1,
    value=5
)

monetary = st.number_input(
    "Monetary Value",
    min_value=0.0,
    value=1000.0
)

total_quantity = st.number_input(
    "Total Quantity Purchased",
    min_value=1,
    value=50
)

average_order_value = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=200.0
)

customer_tenure = st.number_input(
    "Customer Tenure (Days)",
    min_value=0,
    value=180
)

average_purchase_gap = st.number_input(
    "Average Purchase Gap (Days)",
    min_value=0.0,
    value=30.0
)

country = st.text_input(
    "Country",
    value="United Kingdom"
)

# Prediction button
if st.button("Predict Customer Retention"):
    input_data = pd.DataFrame([{
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Total Quantity': total_quantity,
        'Average Order Value': average_order_value,
        'Customer Tenure': customer_tenure,
        'Average Purchase Gap': average_purchase_gap,
        'Country': country
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    inactive_probability = probabilities[1] * 100
    active_probability = probabilities[0] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is predicted to become INACTIVE.")
    else:
        st.success("Customer is predicted to remain ACTIVE.")

    st.write(
        f"**Probability of remaining active:** "
        f"{active_probability:.2f}%"
    )
    st.write(
        f"**Probability of becoming inactive:** "
        f"{inactive_probability:.2f}%"
    )

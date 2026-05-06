import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ========================
# LOAD MODEL
# ========================
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# ========================
# TITLE
# ========================
st.title("🎯 Intelligent Marketing Strategy System")

# ========================
# USER INPUT
# ========================
st.subheader("📥 Input Customer Data")

age = st.slider("Age", 18, 70, 30)
job = st.selectbox("Job", ["admin.", "technician", "services", "management"])
marital = st.selectbox("Marital", ["single", "married", "divorced"])
education = st.selectbox("Education", ["basic.4y", "high.school", "university.degree"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
default = st.selectbox("Default Credit", ["yes", "no"])
campaign = st.slider("Campaign Contacts", 0, 10, 1)
previous = st.slider("Previous Contacts", 0, 10, 0)
poutcome = st.selectbox("Previous Outcome", ["success", "failure", "nonexistent"])

# ========================
# CREATE DATAFRAME
# ========================
input_df = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'housing': housing,
    'loan': loan,
    'default': default,
    'campaign': campaign,
    'previous': previous,
    'poutcome': poutcome
}])

# ========================
# PROCESS BUTTON
# ========================
if st.button("🔍 Analyze Customer"):

    # preprocessing
    X = preprocessor.transform(input_df)
    X = X.toarray()
    X_scaled = scaler.transform(X)

    # clustering
    cluster = kmeans.predict(X_scaled)[0]

    # ========================
    # SCORING SYSTEM
    # ========================
    housing_val = 1 if housing == "yes" else 0
    loan_val = 1 if loan == "yes" else 0
    default_val = 1 if default == "yes" else 0

    # simple normalization (manual)
    campaign_norm = campaign / 10
    previous_norm = previous / 10

    engagement_score = 0.6 * campaign_norm + 0.4 * previous_norm
    risk_score = 0.4 * loan_val + 0.3 * housing_val + 0.3 * default_val

    final_score = 0.6 * engagement_score + 0.4 * (1 - risk_score)

    # ========================
    # CATEGORY
    # ========================
    if final_score > 0.7:
        category = "High Value"
        strategy = "Offer Premium Products & Investment Services"
    elif final_score > 0.4:
        category = "Medium Value"
        strategy = "Cross-Selling Financial Products"
    else:
        category = "Low Value"
        strategy = "Promotional Campaign & Engagement Boost"

    # ========================
    # OUTPUT
    # ========================
    st.subheader("📊 Result")

    st.write(f"Cluster: {cluster}")
    st.write(f"Final Score: {round(final_score,3)}")
    st.write(f"Category: {category}")

    st.subheader("🎯 Recommended Strategy")
    st.success(strategy)
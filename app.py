# 

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

# =========================
# LOAD MODEL
# =========================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
X_scaled = joblib.load("X_scaled.pkl")
clusters_data = joblib.load("clusters.pkl")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("About App")
st.sidebar.info("""
This app uses Machine Learning (K-Means Clustering)
to segment bank customers and recommend marketing strategies.
Upgraded with profiling, explainability, and evaluation metrics.
""")

st.title("Bank Customer Segmentation & Marketing Strategy System 🚀")

# =========================
# CLUSTER LABEL
# =========================
cluster_names = {
    0: "Low Engagement Customers",
    1: "Potential Customers",
    2: "At-Risk Customers",
    3: "High Value Customers"
}

# =========================
# CLUSTER PROFILING
# =========================
def get_cluster_profile(cluster):
    df = pd.DataFrame(X_scaled, columns=features)
    df["cluster"] = clusters_data
    cluster_df = df[df["cluster"] == cluster]

    return {
        "avg_campaign": cluster_df["campaign"].mean(),
        "avg_pdays": cluster_df["pdays"].mean(),
        "avg_euribor": cluster_df["euribor3m"].mean()
    }

# =========================
# STRATEGY (DATA-DRIVEN)
# =========================
def marketing_strategy(profile):
    if profile["avg_campaign"] < 2:
        return "Awareness Strategy: Digital marketing, education content, email campaign"

    elif profile["avg_campaign"] < 5:
        return "Conversion Strategy: Offer credit cards, loans, limited-time promotions"

    elif profile["avg_euribor"] > 3:
        return "High Value Strategy: Investment, deposito, priority banking"

    else:
        return "Retention Strategy: Personalized offers and loyalty programs"

# =========================
# INPUT USER
# =========================
st.markdown("### 📊 Input Customer Data")

age = st.number_input("Age", 18, 100, 30)
campaign = st.number_input("Number of contacts", 1, 50, 2)
pdays = st.number_input("Days since last contact", 0, 999, 999)
previous = st.number_input("Previous contacts", 0, 10, 0)
emp_var_rate = st.number_input("Employment variation rate", -5.0, 5.0, 1.0)
cons_price_idx = st.number_input("Consumer price index", 90.0, 100.0, 93.0)
cons_conf_idx = st.number_input("Consumer confidence index", -60.0, 0.0, -40.0)
euribor3m = st.number_input("Interest rate (Euribor)", 0.0, 6.0, 4.0)
nr_employed = st.number_input("Number of employees", 4000.0, 6000.0, 5200.0)

# =========================
# PREDICTION
# =========================
if st.button("Predict Strategy"):

    input_data = pd.DataFrame([[ 
        age, campaign, pdays, previous,
        emp_var_rate, cons_price_idx, cons_conf_idx,
        euribor3m, nr_employed
    ]], columns=[
        "age","campaign","pdays","previous",
        "emp.var.rate","cons.price.idx","cons.conf.idx",
        "euribor3m","nr.employed"
    ])

    input_full = pd.DataFrame(columns=features)
    input_full.loc[0] = 0

    for col in input_data.columns:
        if col in input_full.columns:
            input_full[col] = input_data[col]

    input_scaled = scaler.transform(input_full)
    cluster = kmeans.predict(input_scaled)[0]

    profile = get_cluster_profile(cluster)
    strategy = marketing_strategy(profile)

    st.success(f"Cluster: {cluster} - {cluster_names[cluster]}")

    st.info(f"""
    📌 Customer Profile :
    - Avg Campaign: {profile['avg_campaign']:.2f}
    - Avg Last Contact: {profile['avg_pdays']:.2f}
    - Avg Interest Rate: {profile['avg_euribor']:.2f}

    🎯 Recommended Strategy:
    {strategy}
    """)

    # =========================
    # EVALUATION
    # =========================
    score = silhouette_score(X_scaled, clusters_data)
    st.write(f"Model Silhouette Score: {score:.3f}")

    # =========================
    # VISUALIZATION
    # =========================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": clusters_data
    })

    user_pca = pca.transform(input_scaled)

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Customer Segmentation (PCA)"
    )

    fig.add_scatter(
        x=[user_pca[0][0]],
        y=[user_pca[0][1]],
        mode="markers+text",
        marker=dict(size=15, symbol="star"),
        name="User",
        text=["YOU"],
        textposition="top center"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# CLUSTER SUMMARY
# =========================
st.markdown("### 📊 Cluster Summary")
summary_df = pd.DataFrame(X_scaled, columns=features)
summary_df["cluster"] = clusters_data
st.dataframe(summary_df.groupby("cluster").mean())

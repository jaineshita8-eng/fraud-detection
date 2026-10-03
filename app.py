import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detector")

st.caption(
    "XGBoost + SMOTE trained on the ULB dataset "
    "(284,807 transactions, 0.17% fraud)"
)


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model.joblib")


@st.cache_data
def load_samples():
    return pd.read_csv(BASE_DIR / "sample_transactions.csv")


model = load_model()
samples = load_samples()

threshold = st.sidebar.slider(
    "Fraud threshold",
    0.05,
    0.95,
    0.50,
    0.05
)


tab1, tab2 = st.tabs(
    ["Try a sample transaction", "Upload CSV"]
)


with tab1:

    idx = st.selectbox(
        "Pick a transaction",
        samples.index
    )

    row = samples.loc[[idx]]

    st.dataframe(row)

    if st.button("Check transaction"):

        prob = model.predict_proba(row)[0, 1]

        if prob >= threshold:
            st.error(
                f"🚨 FRAUD, probability {prob:.1%}"
            )
        else:
            st.success(
                f"✅ Legit, fraud probability {prob:.1%}"
            )


with tab2:

    file = st.file_uploader(
        "Upload CSV with the same columns",
        type="csv"
    )

    if file:

        data = pd.read_csv(file)[samples.columns]

        data["fraud_probability"] = (
            model.predict_proba(data)[:, 1]
        )

        data["prediction"] = (
            data["fraud_probability"] >= threshold
        ).map({
            True: "Fraud",
            False: "Legit"
        })

        st.dataframe(data)
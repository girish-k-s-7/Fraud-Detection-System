import streamlit as st
import pandas as pd
import numpy as np

from src.pipelines.prediction_pipeline import FraudPredictor


st.set_page_config(page_title="Fraud Detection System", layout="centered")


@st.cache_resource
def load_predictor():
    return FraudPredictor()


@st.cache_data
def load_sample_data():
    # Using test data created by the pipeline
    return pd.read_csv("artifacts/fraud_test.csv")


def main():
    st.title("Credit Card Fraud Detection System")
    st.write(
        "This app uses a trained machine learning model to predict whether a "
        "credit card transaction is likely to be **fraudulent**."
    )

    predictor = load_predictor()
    df = load_sample_data()

    st.subheader("Select a Transaction from Test Data")

    idx = st.slider("Choose a row index:", 0, len(df) - 1, 0)
    row = df.iloc[[idx]]
    feature_cols = [col for col in df.columns if col != "Class"]
    features = row[feature_cols]

    st.write("### Transaction Features")
    st.dataframe(features, use_container_width=True)

    if st.button("Predict Fraud"):
        preds, probs = predictor.predict(features)

        fraud_prob = probs[0]
        pred_label = preds[0]

        if pred_label == 1:
            st.error(f"Fraud Detected! (Probability: {fraud_prob:.2%})")
        else:
            st.success(f"Transaction is Legitimate (Fraud Probability: {fraud_prob:.2%})")

         
        actual = int(row["Class"].values[0])
        st.write("**Actual Label:**", "Fraud" if actual == 1 else "Not Fraud")


if __name__ == "__main__":
    main()

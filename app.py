import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
pca = joblib.load("pca.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🏺 Archaeological Artifact Classification")

object_name = st.text_input("Object Name")
culture = st.text_input("Culture")
medium = st.text_input("Medium")
country = st.text_input("Country")

if st.button("Predict"):

    text = object_name + " " + culture + " " + medium + " " + country

    X = tfidf.transform([text])

    X = pca.transform(X.toarray())

    pred = model.predict(X)

    proba = model.predict_proba(X)

    confidence = np.max(proba) * 100

    period = encoder.inverse_transform(pred)

    st.success(f"Predicted Period : {period[0]}")

    st.info(f"Confidence : {confidence:.2f}%")

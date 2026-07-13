import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
pca = joblib.load("pca.pkl")
label_encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Archaeological Artifact Classification")

st.title("🏺 Archaeological Artifact Classification")

st.write("Predict the Cultural Period of an archaeological artifact.")

object_name = st.text_input("Object Name")
culture = st.text_input("Culture")
medium = st.text_input("Medium")
country = st.text_input("Country")

if st.button("Predict"):

    text = object_name + " " + culture + " " + medium + " " + country

    vector = tfidf.transform([text])

    vector = pca.transform(vector.toarray())

    prediction = model.predict(vector)

    result = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Cultural Period: {result[0]}")

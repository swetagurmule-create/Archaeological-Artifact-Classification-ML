import streamlit as st

st.set_page_config(
    page_title="Archaeological Artifact Classification",
    page_icon="🏺",
    layout="centered"
)

st.title("🏺 Archaeological Artifact Classification")

st.write("""
Welcome to the Archaeological Artifact Classification Project.

This project is developed using Machine Learning to classify archaeological artifacts.

Currently this is the deployed version of the project.
The prediction model will be integrated soon.
""")

uploaded_file = st.file_uploader(
    "Upload an Artifact Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")

    if st.button("Predict"):
        st.info("Model integration is under development.")

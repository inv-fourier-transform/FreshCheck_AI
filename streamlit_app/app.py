import streamlit as st
from model_helper import predict_freshness

st.title("Fruits freshness classifier")
st.subheader("Upload an image of the fruit to detect if it is fresh or not.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image_path ="temp_file.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    freshness_prediction_label, confidence = predict_freshness(image_path)
    st.info(f"Prediction: {freshness_prediction_label}")
    st.write(f"Confidence: **{confidence:.2%}**")

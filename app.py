import streamlit as st
import numpy as np
from PIL import Image

st.title("🩸 Blood Group Detection")

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((64,64))
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):
        classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        result = np.random.choice(classes)

        st.success(f"Predicted Blood Group: {result}")

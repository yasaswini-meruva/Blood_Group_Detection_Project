import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("blood_model.h5")
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

st.title("🩸 Blood Group Detection")

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((64,64))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        result = class_names[np.argmax(prediction)]

        st.success(f"Predicted Blood Group: {result}")
import streamlit as st
st.set_page_config(page_title="Teeth Classifier", layout="centered")  # MUST be first Streamlit command

import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
   model = tf.keras.models.load_model(r"C:\Users\ibgam\Documents\GitHub\Teeth-Type-Classification-App\APP\Model\teeth_EffNet_model.h5")
   return model

model = load_model()

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("🦷 Teeth Image Classifier")
st.markdown("Upload an RGB image of a tooth to classify it into one of 7 dental categories.")

uploaded_file = st.file_uploader("Choose a JPG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"🧠 Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2%}")

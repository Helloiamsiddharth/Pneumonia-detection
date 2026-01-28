import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "pneumonia_detection_model.h5"
model = load_model(MODEL_PATH)

def preprocess_image(img_array):
    """Apply classical CV techniques to image array (H, W, C) in [0, 255]."""
    # Convert to uint8
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    
    # Grayscale conversion
    if img_array.shape[-1] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.squeeze()
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)
    
    # Resize
    enhanced_resized = cv2.resize(enhanced, (224, 224))
    edges_resized = cv2.resize(edges, (224, 224))
    
    # To RGB
    enhanced_rgb = cv2.cvtColor(enhanced_resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    return enhanced_rgb.astype(np.float32) / 255.0

def predict(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    processed = preprocess_image(img)
    input_img = np.expand_dims(processed, axis=0)
    
    prob = model.predict(input_img)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    conf = max(prob, 1 - prob) * 100
    
    return label, conf, img

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image (.jpg/.png) to get prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    label, conf, img = predict(uploaded_file)
    
    st.image(img, channels="BGR", caption=f"Prediction: **{label}** ({conf:.2f}%)", use_column_width=True)
    st.success(f"**{label}** with **{conf:.2f}%** confidence")

st.markdown("---")
st.caption("Model: Fine-tuned ResNet-50 | Trained on Kaggle Chest X-Ray Pneumonia dataset")
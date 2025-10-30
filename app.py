# ============================================================
# app.py — Streamlit App for MRI/CT Denoising using U-Net CNN
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import base64

# --------------------- Page Setup --------------------------
st.set_page_config(page_title="Medical Image Denoising", page_icon="🧠")
st.title("🧠 Medical Image Denoising using CNN (U-Net Autoencoder)")
st.write("Upload a noisy **MRI or CT** image (JPG/PNG) and get a denoised version!")

# --------------------- Load Model (from Google Drive) ---------------------
import gdown, os

@st.cache_resource
def load_unet_model():
    model_path = "unet_denoiser_mri_ct.h5"
    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model from Google Drive...")
        # Replace the file ID below with your actual ID
        file_id = "1AbCdEFGhIJklmnopQRsTuvWXyz"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = load_model(model_path, compile=False)
    return model

model = load_unet_model()
st.success("✅ Model loaded successfully!")


# --------------------- Helper Functions ---------------------
def preprocess_image(uploaded_file, size=(128, 128)):
    """Convert uploaded image to grayscale, resize, normalize to [0,1]."""
    img = Image.open(uploaded_file).convert('L')
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # shape (1, 128,128,1)
    return arr, img

def postprocess(pred):
    """Convert predicted [0,1] float image to 8-bit PIL image."""
    arr = np.squeeze(pred) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# --------------------- Upload Section -----------------------
uploaded = st.file_uploader("📤 Upload MRI/CT image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    x, display_img = preprocess_image(uploaded)

    # Show input
    st.image(display_img, caption="Uploaded (Preprocessed)", use_container_width=True)

    # Predict
    with st.spinner("🧠 Denoising... please wait"):
        denoised_pred = model.predict(x, verbose=0)
    denoised_img = postprocess(denoised_pred)

    # Show comparison
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Noisy Input", use_container_width=True)
    with col2:
        st.image(denoised_img, caption="Denoised Output", use_container_width=True)

    # Download option
    buf = BytesIO()
    denoised_img.save(buf, format="PNG")
    st.download_button("📥 Download Denoised Image", buf.getvalue(), "denoised_image.png")

else:
    st.info("👆 Upload an MRI or CT image above to begin denoising.")

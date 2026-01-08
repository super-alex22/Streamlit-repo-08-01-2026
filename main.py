import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9) and the AI will recognize it.")

# --------------------------------------------------
# Load CNN Model
# --------------------------------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("digit_cnn.h5")
        return model
    except Exception as e:
        st.error("❌ Could not load model. Make sure digit_cnn.h5 exists.")
        st.stop()

model = load_cnn_model()
st.success("✅ CNN model loaded successfully!")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # ------------------------------------------
        # Image Preprocessing (MNIST-style)
        # ------------------------------------------
        img_gray = np.array(image.convert("L"))

        # Resize to 28x28
        img_resized = cv2.resize(img_gray, (28, 28))

        # Blur to remove noise
        img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # Threshold (white bg, black digit)
        _, img_thresh = cv2.threshold(
            img_blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Normalize
        img_norm = img_thresh / 255.0

        # Reshape for CNN
        img_input = img_norm.reshape(1, 28, 28, 1)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------
        prediction = model.predict(img_input)
        digit = np.argmax(prediction)

        st.markdown(f"## 🧠 Prediction: **{digit}**")

        # Probabilities
        st.subheader("Confidence")
        for i, prob in enumerate(prediction[0]):
            st.write(f"Digit {i}: {prob:.2%}")

        # Show processed image
        st.subheader("Processed Image (What the model sees)")
        st.image(img_thresh, clamp=True)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

# --------------------------------------------------
# Sidebar Instructions
# --------------------------------------------------
st.sidebar.header("📌 Instructions")
st.sidebar.write("""
**For best results:**
- White background
- Black digit
- One digit only (0–9)
- Digit centered
- Minimal noise

**Model:**
- CNN trained on MNIST
- 28×28 resolution
- ~99% accuracy
""")

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ using Streamlit & TensorFlow")

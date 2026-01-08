import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9) and the AI will recognize it.")

# --------------------------------------------------
# Train model automatically (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    digits = load_digits()

    X = digits.images.reshape(len(digits.images), -1)
    X = X / X.max()
    y = digits.target

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


model = load_model()
st.success("✅ Model trained and loaded!")

# --------------------------------------------------
# Upload image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        # ------------------------------------------
        # Image preprocessing (8×8)
        # ------------------------------------------
        img_gray = image.convert("L")
        img_resized = img_gray.resize((8, 8))

        img_array = np.array(img_resized).astype(np.float32)

        # Auto invert if background is white
        if img_array.mean() > 127:
            img_array = 255 - img_array

        # Normalize
        img_array = img_array / img_array.max()

        img_flat = img_array.flatten().reshape(1, -1)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------
        prediction = model.predict(img_flat)[0]
        probs = model.predict_proba(img_flat)[0]

        st.markdown(f"## 🧠 Prediction: **{prediction}**")

        st.subheader("Confidence")
        for i, prob in enumerate(probs):
            st.write(f"Digit {i}: {prob:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Instructions")
st.sidebar.write("""
- Upload one digit (0–9)
- White background
- Black digit
- Centered digit
- Minimal noise
""")

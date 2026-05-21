import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Satellite Image Classification",
    page_icon="🛰️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Background */

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
    color: white;
}

/* Title */

.main-title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: white;
    margin-top: 10px;
}

/* Subtitle */

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* Upload Box */

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Prediction Card */

.prediction-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
}

/* Footer */

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
}

/* Button */

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("satellite_model.h5")

model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 128

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛰️ Satellite Image Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-Based Land Cover Detection Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(img):

    # Resize image
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    img_array = image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    return prediction, predicted_class, confidence

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    # Safe image loading
    img = Image.open(uploaded_file).convert("RGB")

    # Create columns
    col1, col2 = st.columns([1,1])

    # =====================================================
    # LEFT SIDE - IMAGE
    # =====================================================

    with col1:

        st.image(
            img,
            caption="Uploaded Satellite Image",
            width=350
        )

    # =====================================================
    # RIGHT SIDE - PREDICTION
    # =====================================================

    with col2:

        with st.spinner("Predicting Image..."):

            prediction, predicted_class, confidence = predict_image(img)

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {predicted_class}")

        st.info(f"Confidence Score: {confidence:.2f}%")

        st.progress(int(confidence))

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # =====================================================
    # CONFIDENCE GRAPH
    # =====================================================

    st.subheader("Prediction Confidence")

    confidence_scores = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(class_names, confidence_scores)

    plt.xticks(rotation=45)

    plt.ylabel("Confidence (%)")

    plt.title("Prediction Confidence Scores")

    st.pyplot(fig)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Features

✅ Satellite Image Classification  
✅ Deep Learning Prediction  
✅ Confidence Score  
✅ Confidence Graph  
✅ MobileNetV2 Transfer Learning  

---

### Dataset

EuroSAT RGB Dataset

---

### Technologies Used

- TensorFlow
- Streamlit
- MobileNetV2
- Matplotlib

---

### Model

Transfer Learning Based CNN
""")

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'Developed for Final Year Project | AI-Based Satellite Image Analysis'
    '</div>',
    unsafe_allow_html=True
)

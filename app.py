import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
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

IMG_SIZE = 224

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

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    return prediction, predicted_class, confidence, img_array

# =========================================================
# GRAD-CAM FUNCTION
# =========================================================

def generate_gradcam(img_array):

    # Get base MobileNetV2 model
    base_model = model.layers[0]

    # Last convolution layer
    last_conv_layer = base_model.get_layer("out_relu")

    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # Compute gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    # Get gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Mean intensity of gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Feature map
    conv_outputs = conv_outputs[0]

    # Heatmap
    heatmap = tf.reduce_sum(
        tf.multiply(pooled_grads, conv_outputs),
        axis=-1
    )

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)

    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    # Safe image loading
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    # =====================================================
    # LEFT SIDE - IMAGE
    # =====================================================

    with col1:

        st.image(
            img,
            caption="Uploaded Satellite Image",
            width=400
        )

    # =====================================================
    # RIGHT SIDE - PREDICTION
    # =====================================================

    with col2:

        prediction, predicted_class, confidence, img_array = predict_image(img)

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

    fig, ax = plt.subplots(figsize=(12,5))

    ax.bar(class_names, confidence_scores)

    plt.xticks(rotation=45)

    plt.ylabel("Confidence (%)")

    plt.title("Prediction Confidence Scores")

    st.pyplot(fig)

    # =====================================================
    # GRAD-CAM HEATMAP
    # =====================================================

    st.subheader("Grad-CAM Heatmap")

    try:

        heatmap = generate_gradcam(img_array)

        # Original image
        original = np.array(
            img.resize((IMG_SIZE, IMG_SIZE))
        )

        # Resize heatmap
        heatmap = cv2.resize(
            heatmap,
            (IMG_SIZE, IMG_SIZE)
        )

        # Convert heatmap
        heatmap = np.uint8(255 * heatmap)

        # Apply color map
        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # Overlay
        superimposed_img = heatmap * 0.4 + original

        # Display
        fig2, ax2 = plt.subplots(figsize=(7,7))

        ax2.imshow(
            cv2.cvtColor(
                np.uint8(superimposed_img),
                cv2.COLOR_BGR2RGB
            )
        )

        ax2.axis("off")

        st.pyplot(fig2)

    except Exception as e:

        st.error(
            "Grad-CAM visualization could not be generated."
        )

        st.text(str(e))

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Features

✅ Satellite Image Classification  
✅ Deep Learning Prediction  
✅ Confidence Score  
✅ Grad-CAM Heatmap  
✅ MobileNetV2 Transfer Learning  

---

### Dataset

EuroSAT RGB Dataset

---

### Technologies Used

- TensorFlow
- Streamlit
- OpenCV
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

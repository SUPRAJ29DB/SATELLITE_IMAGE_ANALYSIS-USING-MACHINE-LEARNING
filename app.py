import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(
    page_title="Satellite Image Classification",
    page_icon="🛰️",
    layout="wide"
)

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

/* Main Title */

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

/* Streamlit Button */

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("satellite_model.h5")

model = load_model()

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

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

def predict_image(img):

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    return prediction, predicted_class, confidence, img_array

def display_gradcam(img_array):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.layers[0].get_layer("Conv_1").output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    heatmap = heatmap / np.max(heatmap)

    return heatmap.numpy()

if uploaded_file is not None:

    col1, col2 = st.columns([1,1])

    with col1:

        img = Image.open(uploaded_file).convert("RGB")

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

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

        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Prediction Confidence")

    confidence_scores = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(12,5))

    ax.bar(class_names, confidence_scores)

    plt.xticks(rotation=45)

    plt.ylabel("Confidence (%)")

    plt.title("Prediction Confidence Scores")

    st.pyplot(fig)

    st.subheader("Grad-CAM Heatmap")

    heatmap = display_gradcam(img_array)

    original = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

    heatmap = cv2.resize(
        heatmap,
        (IMG_SIZE, IMG_SIZE)
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = heatmap * 0.4 + original

    fig2, ax2 = plt.subplots(figsize=(7,7))

    ax2.imshow(
        cv2.cvtColor(
            np.uint8(superimposed_img),
            cv2.COLOR_BGR2RGB
        )
    )

    ax2.axis("off")

    st.pyplot(fig2)

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
- Matplotlib
- MobileNetV2

---

### Model
Transfer Learning Based CNN
""")

st.markdown(
    '<div class="footer">'
    'Developed for Final Year Project | AI-Based Satellite Image Analysis'
    '</div>',
    unsafe_allow_html=True
)
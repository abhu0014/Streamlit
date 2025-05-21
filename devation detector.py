import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.title("Dyslexia Deviation Detector")

@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

feature_model = load_model()

def extract_features(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(np.expand_dims(image, axis=0))
    features = feature_model.predict(image, verbose=0)
    return features.flatten()

def extract_shape_features(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
    else:
        x = y = w = h = 0
    aspect_ratio = h / w if w != 0 else 0
    return contour_count, h, w, aspect_ratio

# Upload images
dys_img_file = st.file_uploader("Upload Dyslexic Handwritten Letter Image", type=["png", "jpg", "jpeg"])
norm_img_file = st.file_uploader("Upload Normal Handwritten Letter Image", type=["png", "jpg", "jpeg"])

if dys_img_file and norm_img_file:
    # Read images as numpy arrays
    dys_img = cv2.imdecode(np.frombuffer(dys_img_file.read(), np.uint8), 1)
    norm_img = cv2.imdecode(np.frombuffer(norm_img_file.read(), np.uint8), 1)

    st.image([dys_img[..., ::-1], norm_img[..., ::-1]], caption=["Dyslexic", "Normal"], width=224)

    # Feature extraction
    dyslexic_features = extract_features(dys_img)
    normal_features = extract_features(norm_img)

    similarity_score = cosine_similarity([normal_features], [dyslexic_features])[0][0]
    deviation = 1 - similarity_score

    feature_diff = np.abs(normal_features - dyslexic_features)
    mean_diff = np.mean(feature_diff)
    max_diff = np.max(feature_diff)
    min_diff = np.min(feature_diff)

    dys_contours, dys_h, dys_w, dys_ar = extract_shape_features(dys_img)
    norm_contours, norm_h, norm_w, norm_ar = extract_shape_features(norm_img)
    contour_diff = abs(dys_contours - norm_contours)
    height_diff = abs(dys_h - norm_h)
    width_diff = abs(dys_w - norm_w)
    ar_diff = abs(dys_ar - norm_ar)

    threshold = 0.3
    if deviation > threshold:
        prediction_text = "⚠️ Possibly DYSLEXIC"
    else:
        prediction_text = "✅ Likely NORMAL"

    st.markdown(f"### Prediction: {prediction_text}")
    st.write(f"**Similarity Score:** {similarity_score:.4f}")
    st.write(f"**Deviation:** {deviation:.4f}")
    st.write(f"**Mean Abs Deep Feature Diff:** {mean_diff:.4f}")
    st.write(f"**Max/Min Deep Feature Diff:** {max_diff:.4f} / {min_diff:.4f}")
    st.write(f"**Contour Count (Dys/Norm):** {dys_contours} / {norm_contours} | **Diff:** {contour_diff}")
    st.write(f"**Height (Dys/Norm):** {dys_h} / {norm_h} | **Diff:** {height_diff}")
    st.write(f"**Width (Dys/Norm):** {dys_w} / {norm_w} | **Diff:** {width_diff}")
    st.write(f"**Aspect Ratio (Dys/Norm):** {dys_ar:.2f} / {norm_ar:.2f} | **Diff:** {ar_diff:.2f}")
    st.write(f"**Threshold:** {threshold}")

    # Bar graph for first 40 feature differences
    st.markdown("#### First 40 Deep Feature Differences")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(1, 41), feature_diff[:40])
    ax.set_title("First 40 Deep Feature Differences")
    ax.set_xlabel("Deep Feature Index (1-40)")
    ax.set_ylabel("Absolute Difference (|Normal - Dyslexic|)")
    ax.text(
        0.5, -0.28,
        "Each bar shows the absolute difference for a specific deep feature extracted by VGG16.\n"
        "Bar N = |Feature value (Normal, #N) - Feature value (Dyslexic, #N)|\n"
        "X-axis: Feature index (1 to 40). Y-axis: Absolute Difference.\n"
        "Higher bars mean greater difference for that feature between the two images.",
        ha='center', va='top', transform=ax.transAxes, fontsize=10, color='dimgray'
    )
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(
            f"F{i+1}\n{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=8, rotation=90
        )
    plt.tight_layout()
    st.pyplot(fig)

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------
# Session State (for Clear)
# -------------------------
if "clear" not in st.session_state:
    st.session_state.clear = False

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model("models/MobileNetV2_best.keras")

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Bone Fracture Detection", layout="centered")

# -------------------------
# UI Styling
# -------------------------
st.markdown("""
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1665873311250-428b5c4d0170");
background-size: cover;
}

.title {
    text-align: center;
    color: white;
    font-size: 42px;
    font-weight: bold;
}

.desc {
    text-align: center;
    color: white;
    font-size: 18px;
}

.section-bar {
    background: rgba(255,255,255,0.15);
    box-shadow: 0 0 15px rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 12px;
    backdrop-filter: blur(8px);
    text-align: center;
    color: white;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Glass Button */
div.stButton > button {
    background: rgba(255, 255, 255, 0.15);
    color: white;
    border-radius: 12px;
    padding: 12px 6px;
    border: 1px solid rgba(255,255,255,0.3);
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("<div class='title'>🦴 X-Ray Bone Fracture Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Detecting What Eyes Might Miss.</div>", unsafe_allow_html=True)

# -------------------------
# About Card
# -------------------------
st.markdown("""
<div class='section-bar'>
<h4 style='color:white;'> 🩺 Project Overview</h4>
<p style='color:white;'>
Developed a deep learning-based system to detect bone fractures from X-ray images using CNN and transfer learning, achieving high accuracy and zero missed fracture cases with MobileNetV2.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='section-bar'>
<h4 style='color:white;'> ⚙️ About the Model</h4>
<p style='color:white;'>
MobileNetV2 is a lightweight deep learning model designed for efficient image classification. 
It uses optimized convolution techniques to reduce computation while maintaining high accuracy, making it suitable for fast and real-time applications.
</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Classes
# -------------------------
class_names = ["Fractured", "Normal"]

# -------------------------
# Upload Section
# -------------------------


uploaded_file = st.file_uploader(
    "Upload X-ray",
    type=["jpg","png","jpeg"],
    key=st.session_state.get("uploader_key", "uploader_1")
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn1:
    predict_btn = st.button("🔍 Predict Fracture",
        use_container_width=True, 
        disabled=(uploaded_file is None)
        
    )

with col_btn3:
    clear_btn = st.button("🧹 Clear All", use_container_width=True)
    if clear_btn:
        st.session_state.uploader_key = f"uploader_{np.random.randint(10000)}"    
    
# -------------------------
# Preprocess Function
# -------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------
# Description
# -------------------------
def get_description(label):
    if label == "Fractured":
        return "⚠️ Possible fracture detected. Please consult a doctor."
    else:
        return "✅ Bone looks normal."

# -------------------------
# MAIN DISPLAY
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    display_image = image.resize((280, 280))

    if predict_btn:
        
        st.markdown("""
        <div style="display:flex; gap:10px;">
            <div class='section-bar' style='width:50%;'>🦴 Image</div>
            <div class='section-bar' style='width:50%;'>📈 Prediction</div>
        </div>
        """, unsafe_allow_html=True)      
        

        col1, col2 = st.columns(2)

        # LEFT → IMAGE
        with col1:
            st.image(display_image, width=280)

        # RIGHT → RESULT
        with col2:
            with st.spinner("🧠 Analyzing X-ray..."):
                img_array = preprocess_image(image)
                preds = model.predict(img_array)

            # ✅ Correct sigmoid logic
            prob = preds[0][0]

            if prob > 0.5:
                predicted_class = "Normal"
                confidence = prob * 100
            else:
                predicted_class = "Fractured"
                confidence = (1 - prob) * 100

            # Glass Box
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.18);
                padding:20px;
                border-radius:15px;
                text-align:center;
                color:white;">
                <h2>{predicted_class}</h2>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(get_description(predicted_class))

        # -------------------------
        # Precautions
        # -------------------------
        st.markdown("### 🚑 Precautions")


        if predicted_class == "Fractured":
            # Using 'error' gives a red background (urgent)
            st.error("### ⚠️ Fracture Detected")
            
            with st.expander("📋 Immediate Action Plan", expanded=True):
                st.write("""
                * 🏥 **Professional Care:**  Seek medical help immediately for an X-ray or cast.
                * 🚫 **Immobilize:** Avoid all movement of the affected area.
                * 🛌 **Total Rest:** Keep the injured limb elevated and supported.
                * ❄️ **Ice Therapy:** Apply ice packs to reduce pain and swelling.
                """)
                
        else:
            # Using 'success' gives a green background (positive)
            st.success("### ✅ Bone Appears Intact")
            
            with st.expander("🛡️ Preventive Bone Care Tips", expanded=True):
                st.write("""
                * 🍚 **Nutrition:** Increase intake of Vitamin D and Calcium (Milk, Leafy Greens).
                * 🏃🏻‍♂️ **Activity:** Incorporate weight-bearing exercises to build bone density.
                * 🚴🏻‍♀️ **Lifestyle:** Avoid smoking and excessive alcohol, which weaken bones.
                """)
                

st.markdown("""
<div style="text-align:center; color:white; font-size:14px; opacity:0.8;">
    👩‍💻 Developed by <b>Shravani More</b> | 🚀 AI & ML Enthusiast
</div>
""", unsafe_allow_html=True)
                
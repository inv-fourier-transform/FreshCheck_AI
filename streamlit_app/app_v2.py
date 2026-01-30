import streamlit as st
from model_helper import predict_freshness
import os

# Page configuration with custom title and icon
st.set_page_config(
    page_title="FreshCheck AI",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
custom_css = """
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .title-text {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }

    .subtitle-text {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .upload-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title-text">🍎 FreshCheck AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a fruit image and our AI will detect its freshness</p>',
            unsafe_allow_html=True)

# Upload section
with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        uploaded_image = st.file_uploader(
            "📤 Drop your image or click to browse",
            type=["jpg", "jpeg", "png"]
        )

if uploaded_image:
    img_col, res_col = st.columns([1, 1], gap="large")

    with img_col:
        st.markdown("### 📷 Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    with st.spinner("🔍 Analyzing..."):
        label, confidence = predict_freshness(image_path)

    with res_col:
        st.markdown("### 🎯 Results")

        is_fresh = "fresh" in label.lower()
        status_color = "green" if is_fresh else "red"
        emoji = "✅" if is_fresh else "⚠️"

        # Styled result container
        st.markdown(f'''
        <div style="background: white; border-radius: 15px; padding: 1.5rem; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
                    border-left: 5px solid {status_color};">
            <h4>{emoji} Prediction</h4>
            <h2 style="color: {status_color};">{label}</h2>
            <p>Confidence: <b>{confidence:.1%}</b></p>
        </div>
        ''', unsafe_allow_html=True)

        # Progress bar for confidence
        # st.progress(confidence, text="Confidence Score")

    # Tips expander
    with st.expander("💡 Tips for better results"):
        st.markdown("- Use good lighting  \n- Center the fruit  \n- Avoid shadows")

    os.remove(image_path)

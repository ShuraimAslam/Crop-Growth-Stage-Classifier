import streamlit as st
from PIL import Image
from model import predict_stage
from utils import format_confidence

def main():
    # Page configuration
    st.set_page_config(
        page_title="Crop Growth Stage Classifier",
        page_icon="🌱",
        layout="centered"
    )

    # Title and Description
    st.title("🌱 Crop Growth Stage Classifier")
    st.markdown("""
    Welcome to the AI-powered Crop Growth Stage Classifier! 
    This application uses a Vision Transformer (ViT) model to identify the growth stage of your crops from an image.
    
    **Supported Growth Stages:**
    1. Seed Stage
    2. Seedling Stage
    3. Vegetative Growth Stage
    4. Mature Plant Stage
    """)

    st.divider()

    # Image Uploader
    uploaded_file = st.file_uploader(
        "Choose a crop image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        # UI Columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width="stretch")
        
        with col2:
            st.subheader("Classification")
            if st.button("Predict Stage", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Run prediction
                        stage, confidence = predict_stage(image)
                        
                        # Display Results
                        st.success(f"**Predicted Stage:** {stage}")
                        st.write(f"**Confidence:** {format_confidence(confidence)}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                        # Additional context
                        st.info("The classification is based on visual patterns identified by the ViT model.")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

    # Footer
    st.divider()
    st.caption("Powered by Hugging Face Transformers & Streamlit")

if __name__ == "__main__":
    main()

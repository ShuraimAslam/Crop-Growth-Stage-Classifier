import streamlit as st  # The framework for building the web application interface
from PIL import Image  # Library for opening and manipulating images
from model import predict_stage  # Function to run the AI classification logic
from utils import format_confidence  # Utility to format the confidence score

def main():
    # Page configuration: Sets the browser tab title, icon, and overall layout
    st.set_page_config(
        page_title="Crop Growth Stage Classifier",
        page_icon="🌱",
        layout="centered"
    )

    # Title and Description: Displays the main heading and introductory text
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

    st.divider()  # Visual horizontal line separator

    # Image Uploader: Creates a widget for the user to drag and drop or browse files
    uploaded_file = st.file_uploader(
        "Choose a crop image...", 
        type=["jpg", "jpeg", "png"]
    )

    # If an image has been uploaded, proceed with display and classification
    if uploaded_file is not None:
        # Open the uploaded image and convert it to RGB color format
        image = Image.open(uploaded_file).convert("RGB")
        
        # UI Columns: Creates two side-by-side columns for a clean layout
        col1, col2 = st.columns([1, 1])
        
        # Column 1: Displays the image being processed
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width="stretch")  # width="stretch" is deprecated, use_column_width=True is safer
        
        # Column 2: Contains the classification button and displays results
        with col2:
            st.subheader("Classification")
            # When the "Predict Stage" button is clicked...
            if st.button("Predict Stage", type="primary"):
                # Display a loading spinner while the model runs in the background
                with st.spinner("Analyzing image..."):
                    try:
                        # Call the prediction function and unpack the returned stage and confidence
                        stage, confidence = predict_stage(image)
                        
                        # Display Results: Shows the final classification in a success box
                        st.success(f"**Predicted Stage:** {stage}")
                        st.write(f"**Confidence:** {format_confidence(confidence)}")
                        
                        # Progress bar: Visually represents the confidence score (0.0 to 1.0)
                        st.progress(confidence)
                        
                        # Additional context for the user
                        st.info("The classification is based on visual patterns identified by the ViT model.")
                    except Exception as e:
                        # Catch and display any errors that occur during the process
                        st.error(f"An error occurred during prediction: {e}")

    # Footer section
    st.divider()
    st.caption("Powered by Hugging Face Transformers & Streamlit")

# Logic to run the application when the script is executed
if __name__ == "__main__":
    main()

import torch  # Library for deep learning operations
from PIL import Image  # Module for image handling
from transformers import AutoImageProcessor, AutoModelForImageClassification  # Tools for loading pretrained models
from utils import get_stage_name  # Helper to map indices to stage names

# Constants
# We use a standard Vision Transformer (ViT) model from Google
MODEL_NAME = "google/vit-base-patch16-224"

# Global variables for caching the model and processor
# Caching avoids reloading the model every time the Streamlit app reruns
_processor = None
_model = None

def load_model():
    """
    Load the pretrained ViT model and processor from Hugging Face.
    Caches the objects to avoid reloading across streamlit reruns.
    """
    global _processor, _model
    # Check if the model or processor is already loaded in memory
    if _processor is None or _model is None:
        print(f"Loading model and processor from {MODEL_NAME}...")
        # Load the image processor (handles resizing, normalization, etc.)
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        
        # Determine if a GPU (CUDA) is available, otherwise fall back to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the classification model
        # low_cpu_mem_usage=False ensures compatibility on older systems
        _model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, 
            low_cpu_mem_usage=False
        )
        # Move the model to the detected device (GPU or CPU)
        _model.to(device)
        # Set the model to evaluation mode (stops training-specific behaviors like dropout)
        _model.eval()
        
    return _processor, _model

def predict_stage(image: Image):
    """
    Preprocess the image, run inference, and return the predicted stage and confidence.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        tuple: (stage_name, confidence_score)
    """
    # Ensure the model is loaded and ready
    processor, model = load_model()
    # Get the device the model is currently residing on
    device = next(model.parameters()).device

    # Preprocess image: scale, normalize, and convert to a PyTorch tensor
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Perform Inference (Forward Pass)
    with torch.no_grad():  # Disable gradient tracking to save memory and speed up
        outputs = model(**inputs)
        logits = outputs.logits  # Raw scores from the last layer of the model

    # Apply softmax to convert raw scores (logits) into probabilities (0 to 1)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Identify the class with the highest probability (confidence) and its index
    conf, index = torch.max(probs, dim=-1)
    
    # Map the predicted index to our 4 crop growth stages using modulo 4
    # Note: In a real fine-tuned model, it would have exactly 4 output nodes.
    predicted_index = index.item()
    stage_name = get_stage_name(predicted_index)
    confidence_score = conf.item()

    return stage_name, confidence_score

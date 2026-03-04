import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from utils import get_stage_name

# Constants
MODEL_NAME = "google/vit-base-patch16-224"

# Global variables for caching the model and processor
_processor = None
_model = None

def load_model():
    """
    Load the pretrained ViT model and processor from Hugging Face.
    Caches the objects to avoid reloading across streamlit reruns.
    """
    global _processor, _model
    if _processor is None or _model is None:
        print(f"Loading model and processor from {MODEL_NAME}...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        
        # Use a more robust loading approach
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and ensure it's materialized on the correct device
        # We explicitly set low_cpu_mem_usage=False to avoid meta tensor issues
        # on systems where it might be triggered unexpectedly.
        _model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, 
            low_cpu_mem_usage=False
        )
        _model.to(device)
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
    processor, model = load_model()
    device = next(model.parameters()).device

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the top class and confidence
    conf, index = torch.max(probs, dim=-1)
    
    # Since the base model is not explicitly fine-tuned for the 4 stages,
    # we map the predicted index modulo 4 to one of the stages.
    # In a production app, the model would have 4 output nodes specifically for these stages.
    predicted_index = index.item()
    stage_name = get_stage_name(predicted_index)
    confidence_score = conf.item()

    return stage_name, confidence_score

"""
Utility functions for crop growth stage classification.
"""

# Mapping of class indices to human-readable growth stages.
# Note: Since we are using a general-purpose pretrained model (vit-base-patch16-224),
# these mappings are illustrative for the application's structure.
# In a real-world scenario, the model would be fine-tuned on specific crop data.
STAGE_MAPPING = {
    0: "Seed Stage",
    1: "Seedling Stage",
    2: "Vegetative Growth Stage",
    3: "Mature Plant Stage"
}

def get_stage_name(index: int) -> str:
    """
    Map an integer index to a growth stage name.
    If the index is out of range, default to 'Unknown Stage'.
    """
    return STAGE_MAPPING.get(index % 4, "Unknown Stage")

def format_confidence(confidence: float) -> str:
    """
    Format confidence score as a percentage string.
    """
    return f"{confidence * 100:.2f}%"

# Crop Growth Stage Classifier

An AI-powered web application built with Streamlit and Hugging Face Transformers to classify crop image growth stages using the Vision Transformer (ViT) model.

## Features

- **AI Classification**: Uses `google/vit-base-patch16-224` for image understanding.
- **Support for 4 Stages**: Seed, Seedling, Vegetative, and Mature Plant stages.
- **Intuitive UI**: Smooth image upload and real-time prediction visualization.
- **Confidence Scoring**: Displays classification confidence with a progress bar.

## Project Structure

```
crop-growth-classifier/
│
├── app.py           # Streamlit UI implementation
├── model.py         # Hugging Face model loading and inference
├── utils.py         # Helper functions and stage mappings
├── requirements.txt # Project dependencies
└── README.md        # Documentation
```

## Setup and Installation

### 1. Clone or Copy the Files
Ensure all project files are in the same directory: `crop-growth-classifier/`.

### 2. Install Dependencies
It is recommended to use a virtual environment. Run:

```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit server:

```bash
streamlit run app.py
```

## Usage
1. Open the application in your browser (usually at `http://localhost:8501`).
2. Upload an image of a crop in one of the growth stages.
3. Click **Predict Stage**.
4. View the predicted growth stage and the model's confidence score.

## Technical Details
- **Architecture**: Vision Transformer (ViT)
- **Frameworks**: PyTorch, Transformers, Streamlit
- **Model**: `google/vit-base-patch16-224`

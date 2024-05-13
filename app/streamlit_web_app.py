import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from utils import *

import torch
from torch import nn
import sys
sys.path.append('../')
from src import model as nn_model

def save_image_or_file(uploaded_file, arg = 'file'):
    data_path = Path("test_images/")
    custom_image_path = data_path / 'test_image.png'    
    with open(custom_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    custom_image_transformed = load_and_preprocess_image(custom_image_path)

    return custom_image_transformed, custom_image_path
    

def make_prediction(model, model_info, uploaded_file, arg = 'file'):

    custom_image_transformed, custom_image_path = save_image_or_file(uploaded_file, arg)

    # Load the model
    model.load_state_dict(model_info)
    model.eval()

    # Predict the label for the image
    # Define the class names from 'A' to 'Z'
    letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    # Additional class names: 'del', 'nothing', 'space'
    additional_classes = ['del', 'nothing', 'space']

    # Concatenate all class names
    class_names = np.array(letters + additional_classes)
    predicted_label, image_pred_probs = predict_image(model,
                                                        custom_image_transformed,
                                                        class_names)

    # Prediction result section
    st.markdown(
        f'<h3 style="color: green;">Prediction Result</h3>', 
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 3])

    # Display prediction label and confidence rate on the left column
    col1.write(f"Predicted Sign: **{predicted_label[0]}**")
    col1.write(f"Confidence: **{image_pred_probs.max()* 100:.2f}%**")

    # Display the uploaded image on the right column
    with col2:
        image = Image.open(custom_image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    os.remove(custom_image_path)

def main():
    st.title("American Sign Language Detector")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Trained Model
    MODEL_LOAD_PATH = "../model/trained_asl_detection_model.pth"
    model_info = torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu'))

    model = nn_model.EfficientNetB0(num_classes=29).to(device)

    st.subheader("Capture Live Image")
    # Take a live picture
    picture = st.camera_input('')

    if picture:
        make_prediction(model, model_info, picture, 'pic')
            
if __name__ == "__main__":
    main()
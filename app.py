import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Load the trained model (adjust the path as needed)
model = load_model('signature_model.h5')

# Title of the app
st.title('Handwritten Signature Recognition System')

# Description
st.markdown("""
    This application allows you to upload a signature image and recognize it using a pre-trained model.
    Simply upload an image of a handwritten signature, and our model will classify it.
""")

# Function to process the uploaded image
def process_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((200, 100))  # Resize to match the model's expected input size
    image_array = np.array(image)  # Convert image to numpy array
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# File uploader widget for signature image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Signature", use_column_width=True)
    st.write("")
    
    # Process the image
    image = Image.open(uploaded_file)
    processed_image = process_image(image)
    
    # Prediction
    prediction = model.predict(processed_image)
    
    # Display result
    st.write(f"Prediction: {np.argmax(prediction)}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
    
    # Display more details about prediction
    st.markdown("### Prediction Details")
    st.write("This is the result of signature recognition. The model is trained to identify different handwritten signatures.")
else:
    st.write("Upload a signature image to get started.")

# Footer
st.markdown("""
    **Handwritten Signature Recognition System** built with Streamlit and TensorFlow.
    """)

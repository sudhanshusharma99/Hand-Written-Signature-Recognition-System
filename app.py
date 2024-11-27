import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load models
modeln = load_model('name.h5')
modelv = load_model('verify.h5')

NAME_CLASSES = [
    'Aaditya', 'Abhay', 'Ajay', 'Aman', 'Amitabh', 'Anuj', 'Arvind', 'Asif',
    'Bala', 'Bhavya', 'Chinmay', 'David', 'Dinesh', 'Durga', 'Gauri', 'Gautam',
    'Hemang', 'Jinesh', 'Junaid', 'Kalpana', 'Kapil', 'Karan', 'Kushi', 'Lalit',
    'Love', 'M. Adnan', 'Mahipal', 'Manish', 'Meera', 'Neeta', 'Niket',
    'Nirmala', 'Parmod', 'Pawan', 'Rahul', 'Raju', 'Ram', 'Ravi', 'Riya',
    'Rudra', 'Shiv', 'Shivam', 'Sudhanshu', 'Sunil', 'Sunita', 'Tanmay',
    'Tilak', 'Utsav', 'Vaibhav', 'Yashwant'
]
VERIFY_CLASSES = ['Forged', 'Real']

# Streamlit app title
st.title("Signature Recognition, Verification, and Validation")

# File uploader
file = st.file_uploader("Please upload a signature image", type=["jpeg", "jpg", "png"])


def preprocess_image(img):
    """
    Preprocess the image for model prediction.
    """
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    img_array = preprocess_input(img_array)  # Normalize for VGG16
    return np.expand_dims(img_array, axis=0)


def extract_signature(image):
    """
    Extract signature region from the uploaded image using color segmentation and contours.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 38, 0])  # Adjust for the target signature color
    upper = np.array([300, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Apply the mask and find contours
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask == 0] = (255, 255, 255)  # Set non-signature region to white

    # Find contours
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) > 0:
        cnts = np.concatenate(cnts)
        x, y, w, h = cv2.boundingRect(cnts)
        ROI = result[y:y + h, x:x + w]
        resized = cv2.resize(ROI, (224, 224))
        return resized
    else:
        return None


if file:
    # Open the uploaded file
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Extract signature
    extracted_signature = extract_signature(image_cv)

    if extracted_signature is not None:
        st.image(extracted_signature, caption="Extracted Signature", use_column_width=True)

        # Preprocess the extracted signature for model prediction
        processed_signature = preprocess_image(Image.fromarray(extracted_signature))

        if st.button("Reveal Name"):
            predictions = modeln.predict(processed_signature)
            predicted_name = NAME_CLASSES[np.argmax(predictions)]
            st.success(f"Model predicts the signature belongs to **{predicted_name}**.")

        if st.button("Validate Signature"):
            predictions = modelv.predict(processed_signature)
            validation_result = VERIFY_CLASSES[np.argmax(predictions)]
            st.success(f"Model validates the signature as **{validation_result}**.")
    else:
        st.error("No signature detected in the image. Please upload a clearer image.")
else:
    st.info("Please upload an image file.")

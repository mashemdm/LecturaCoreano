import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.title("Reconocimiento óptico de Caracteres (OCR)")

# Option for camera input (uncomment if needed)
# img_file_buffer = st.camera_input("Toma una Foto")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])


with st.sidebar:
    filtro = st.radio("Aplicar Filtro", ('Con Filtro', 'Sin Filtro'))


if uploaded_file is not None:
    # To read image file buffer with OpenCV:
    img_bytes = uploaded_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if filtro == 'Con Filtro':
        cv2_img = cv2.bitwise_not(cv2_img)  # Apply inversion filter

    # Preprocessing for Korean characters (optional)
    # You can add additional preprocessing steps here if needed,
    # such as grayscale conversion, noise reduction, etc.

    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Ensure Tesseract is configured for Korean recognition
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with your Tesseract path (adjust accordingly)
    config = '-l kor'  # Specify Korean language code

    text = pytesseract.image_to_string(img_rgb, config=config)
    st.write(text)

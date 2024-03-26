import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.title("Reconocimiento óptico de Caracteres (OCR)")

# Image input options
image_source = st.radio("Selecciona origen de la imagen:", ("Cámara", "Subir Archivo"))

if image_source == "Cámara":
    img_file_buffer = st.camera_input("Toma una Foto")
else:
    uploaded_file = st.file_uploader("Sube una Imagen", type=["jpg", "jpeg", "png"])

with st.sidebar:
    filtro = st.radio("Aplicar Filtro", ('Con Filtro', 'Sin Filtro'))

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for better noise handling
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    return thresh

if image_source == "Cámara":
    img_bytes = img_file_buffer.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
elif uploaded_file is not None:
    img_bytes = uploaded_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if cv2_img is not None:  # Check if image is loaded successfully
    if filtro == 'Con Filtro':
        cv2_img = preprocess_image(cv2_img)
    else:
        cv2_img = cv2_img

    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    # Specify Korean language for Tesseract OCR engine
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update Tesseract path if needed
    text = pytesseract.image_to_string(img_rgb, lang='kor')  # Use 'kor' for Korean
    st.write(text)
else:
    st.write("Error: Imagen no cargada.")

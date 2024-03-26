import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

st.title("Reconocimiento óptico de Caracteres")


# img_file_buffer = st.camera_input("Toma una Foto")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

with st.sidebar:
      filtro = st.radio("Aplicar Filtro",('Con Filtro', 'Sin Filtro'))


if uploaded_file is not None:
    # To read image file buffer with OpenCV:
    img_bytes = uploaded_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

      
    if filtro == 'Con Filtro':
         cv2_img=cv2.bitwise_not(cv2_img)
        
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # text=pytesseract.image_to_string(img_rgb)
    text = pytesseract.image_to_string(img_rgb, lang='kor', config=tessdata_dir_config)  
    st.write(text) 
    

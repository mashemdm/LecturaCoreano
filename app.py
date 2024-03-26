import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.title("Reconocimiento óptico de Caracteres (OCR)")

# Improved code for Korean character recognition:
def recognize_korean(img):
  """Recognizes Korean characters in the given image.

  Args:
      img: The image as a NumPy array.

  Returns:
      The recognized text as a string.
  """
  # Install the Korean Tesseract OCR engine if not already installed
  # You can run `pip install pytesseract-lang-kor` in your terminal
  try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path if needed
  except:
    st.error("Error: Korean Tesseract OCR engine not found. Install pytesseract-lang-kor to enable Korean recognition.")
    return ""

  # Convert to grayscale and apply adaptive thresholding for better OCR results
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

  # Recognize text with Korean language code ('kor')
  text = pytesseract.image_to_string(thresh, config='--psm 6 --oem 1 -l kor')
  return text

# User interface elements
# img_file_buffer = st.camera_input("Toma una Foto")  # Commented out for now
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])


with st.sidebar:
  filtro = st.radio("Aplicar Filtro", ('Con Filtro', 'Sin Filtro'))

if uploaded_file is not None:
  # To read image file buffer with OpenCV:
  img_bytes = uploaded_file.read()
  nparr = np.frombuffer(img_bytes, np.uint8)
  cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if filtro == 'Con Filtro':
    cv2_img = cv2.bitwise_not(cv2_img)
  else:
    cv2_img = cv2_img

  # Convert to RGB and recognize Korean characters
  img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
  text = recognize_korean(img_rgb)
  st.write(text)

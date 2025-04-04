import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO
model = YOLO("trained_YOLO_model.pt")
st.title("Playing Card Detection Using YOLOv8")
st.header("Upload an Image to Detect Playing Cards")
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    predictions = model(image)
    result = predictions[0]
    st.header("Detection Results")
    st.image(result.plot(), caption="Detected Playing Cards", use_column_width=True)

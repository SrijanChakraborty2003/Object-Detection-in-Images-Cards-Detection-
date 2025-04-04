import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO

model = YOLO("trained_YOLO_model.pt")
# Title for the web app
st.title("Playing Card Detection Using YOLOv8")

# Upload Image
st.header("Upload an Image to Detect Playing Cards")
image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Load and display the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    predictions = model(image)

    # Access the first result (in case there are multiple predictions)
    result = predictions[0]

    # Display the result with bounding boxes and labels
    st.header("Detection Results")
    st.image(result.plot(), caption="Detected Playing Cards", use_column_width=True)

    # Display additional information (confidence scores and classes)
    # st.write("### Detected Cards Info")
    # card_info = ""
    # for i, (class_id, conf, xyxy) in enumerate(zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)):
    #     card_info += f"Card {i+1} - Class: {model.names[class_id]} | Confidence: {conf:.2f} | Bounding Box: {xyxy.tolist()}<br>"
    
    # st.markdown(card_info, unsafe_allow_html=True)

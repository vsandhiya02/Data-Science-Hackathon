import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained model
model = load_model('fashion_mnist.keras')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("🧥👕 Fashion MNIST Classifier")
st.write("Draw or upload a 28x28 image of a clothing item (like a shoe or shirt)")

# Sidebar options
mode = st.sidebar.radio("Select Input Mode", ['Draw', 'Upload'])

if mode == 'Draw':
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.resize(img.astype('uint8'), (28, 28))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_norm = img_gray / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)

        if st.button("Predict"):
            pred = model.predict(img_input)
            pred_class = np.argmax(pred)
            st.write(f"### Prediction: {class_names[pred_class]}")
            st.bar_chart(pred[0])

elif mode == 'Upload':
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        img_array = np.array(image) / 255.0
        st.image(image, caption="Uploaded Image", width=140)

        img_input = img_array.reshape(1, 28, 28, 1)

        if st.button("Predict"):
            pred = model.predict(img_input)
            pred_class = np.argmax(pred)
            st.write(f"### Prediction: {class_names[pred_class]}")
            st.bar_chart(pred[0])

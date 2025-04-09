import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import base64
from streamlit_drawable_canvas import st_canvas
import io

# Load model
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cnn_model()

# UI
st.title("🧠 Digit Prediction App")
st.subheader("Upload an image or draw a digit")

option = st.radio("Choose input method", ("Upload Image", "Draw Digit"))

image = None
image_data = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)  # to match white-digit-on-black bg
        image = image.resize((28, 28))
        image_data = image
        st.image(image, caption="Uploaded Image", width=150)

else:
    canvas_result = st_canvas(
        fill_color="black",  # Fill color
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
        image = image.resize((28, 28))
        image_data = image
        st.image(image, caption="Drawn Image", width=150)

# Predict button
if image_data and st.button("Predict"):
    try:
        img_array = np.array(image_data).reshape(28, 28, 1).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction, axis=1)[0]) * 100

        st.success(f"Predicted digit: **{predicted_class}** with **{confidence:.2f}%** confidence.")
        st.json({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "raw_prediction": prediction.tolist()
        })

    except Exception as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import numpy as np
from PIL import Image
import keras
from keras.applications.resnet50 import preprocess_input


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="COVID-19 Chest X-Ray Classifier",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_covid_model():

    model = keras.models.load_model(
        "models/covid_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )

    return model


# Load model
model = load_covid_model()


# ============================================================
# CLASS NAMES
# ============================================================

# IMPORTANT:
# These must match the order used when you trained your model.
CLASS_NAMES = [
    "COVID-19",
    "Normal",
    "Viral Pneumonia"
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to the input size expected by ResNet50
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # ResNet50 preprocessing
    image_array = preprocess_input(image_array)

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_class_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_class_index]

    confidence = probabilities[predicted_class_index] * 100

    return predicted_class, confidence, probabilities


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🩺 COVID-19 Chest X-Ray Classifier")

st.write(
    "Upload a chest X-ray image and the trained ResNet50 model "
    "will classify it into one of the available categories."
)

st.warning(
    "⚠️ This application is for educational/demo purposes only "
    "and should not be used for medical diagnosis."
)


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display image
    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Uploaded chest X-ray",
        use_container_width=True
    )

    # Prediction button
    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Analyzing image..."):

            try:

                predicted_class, confidence, probabilities = predict_image(
                    image
                )

                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                st.subheader("Prediction")

                st.success(
                    f"Prediction: **{predicted_class}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                # --------------------------------------------
                # CLASS PROBABILITIES
                # --------------------------------------------

                st.subheader("Class Probabilities")

                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities
                ):

                    probability_percentage = probability * 100

                    st.write(
                        f"**{class_name}: "
                        f"{probability_percentage:.2f}%**"
                    )

                    st.progress(
                        float(probability)
                    )

            except Exception as e:

                st.error(
                    "An error occurred while making the prediction."
                )

                st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Model Information"):

    st.write(
        """
        **Model:** ResNet50 Transfer Learning

        **Input size:** 224 × 224 × 3

        **Output classes:**
        - COVID-19
        - Normal
        - Viral Pneumonia

        **Framework:** TensorFlow / Keras

        **Application:** Educational chest X-ray image classification
        """
    )
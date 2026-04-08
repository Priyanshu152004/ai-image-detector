import streamlit as st
from PIL import Image
import numpy as np
from huggingface_hub import from_pretrained_keras
import tensorflow as tf

# Load model (cached for speed)
@st.cache_resource
def load_model():
    model = from_pretrained_keras("poojakabber1997/ResNetDallE2Fakes")
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = load_model()

# Prediction function
def get_prediction(image):
    img = Image.fromarray(image.astype('uint8'), 'RGB').resize((180, 180))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    real_prob = prediction[0][0]
    fake_prob = 1 - real_prob

    if real_prob > fake_prob:
        return "✅ Real Human Face", real_prob
    else:
        return "⚠️ AI Generated Face", fake_prob

# Streamlit UI
st.title("🧠 AI vs Real Face Detection")
st.write("Upload an image to check whether it is AI-generated or real.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    result, confidence = get_prediction(img_array)

    if "AI" in result:
        st.error(f"{result} (Confidence: {confidence:.2f})")
    else:
        st.success(f"{result} (Confidence: {confidence:.2f})")
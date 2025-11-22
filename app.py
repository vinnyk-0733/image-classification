import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Set page config — MUST BE FIRST Streamlit command
st.set_page_config(page_title="Image Classification App", page_icon="🎯")

# -------------------------------------
# ⚙️ 1. Load the Trained Model
# -------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("image_classification.keras")  # replace with your model file name
    return model

model = load_model()

# -------------------------------------
# 🏷️ 2. Define Class Labels
# -------------------------------------
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -------------------------------------
# 🧩 3. Preprocess Function
# -------------------------------------
def preprocess_image(img):
    img = img.resize((32, 32))  # adjust size if your model uses different input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------------
# 🖼️ 4. Streamlit UI
# -------------------------------------
st.title("🎯 Image Classification App")
st.write("Upload an image and let the trained model predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Classifying... ⏳")
    
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    confidence = np.max(predictions) * 100
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    
    if confidence >= 75:
        st.success(f"✅ Predicted Class: **{predicted_class}** ({confidence:.2f}% confidence)")
    else:
        st.warning(f"⚠️ Confidence: {confidence:.2f}%\n\nThis image does **not belong** to the known classes.")
    
    st.subheader("Prediction Probabilities:")
    st.bar_chart(predictions[0])

else:
    st.info("👆 Upload an image file to start classification.")

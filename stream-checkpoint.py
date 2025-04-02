import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('lung_cancer_model.h5')

# Define class labels
class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Malignancy mapping
malignancy_status = {
    'Adenocarcinoma': 'Malignant',
    'Large Cell Carcinoma': 'Malignant',
    'Normal': 'Benign (No Cancer)',
    'Squamous Cell Carcinoma': 'Malignant'
}

def predict_image(img_data):
    img = image.load_img(img_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of max probability
    probability = prediction[0][predicted_class] * 100  # Convert to percentage
    return class_labels[predicted_class], probability, prediction[0]

# Streamlit UI with enhanced design
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 Lung Cancer Detection from Medical Images")
st.markdown("Upload a medical image to check for lung cancer classification.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = BytesIO(uploaded_file.getbuffer())
    st.image(image_data, caption="🖼️ Uploaded Image", use_column_width=True,width=200)
    
    result, confidence, all_probs = predict_image(image_data)
    malignancy = malignancy_status[result]
    
    st.subheader("🔍 Prediction Results")
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Malignancy Status:** {malignancy}")
    
    # Display result message with better UI feedback
    if result == "Normal":
        st.success("✅ No malignant features detected. You are cancer-free.")
    else:
        st.error(f"⚠️ {result} detected. This is classified as {malignancy}. Further medical evaluation is recommended.")
    
    # Plot probability distribution with seaborn styling
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=class_labels, y=all_probs * 100, palette=["blue", "red", "green", "orange"], ax=ax)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Confidence Distribution")
    plt.xticks(rotation=20)
    st.pyplot(fig)

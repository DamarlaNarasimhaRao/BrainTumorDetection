# File: streamlit_app.py

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from fpdf import FPDF
import io
from booking import generate_google_calendar_link
from datetime import datetime, timedelta


# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model_final.h5")

# Constants
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMG_SIZE = (150, 150)
LAST_CONV_LAYER_NAME = "Conv_1_bn"

# --- Utility Functions ---

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img: Image.Image, model: tf.keras.Model) -> tuple:
    img_array = preprocess_image(img)
    prediction = model.predict(img_array, verbose=0)
    confidence = np.max(prediction)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    return predicted_class, confidence, img_array

def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = LAST_CONV_LAYER_NAME) -> np.ndarray:
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_pil: Image.Image, heatmap: np.ndarray) -> Image.Image:
    img = np.array(img_pil.resize(IMG_SIZE))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + img
    return Image.fromarray(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))

def generate_pdf_report(predicted_class: str, confidence: float, filename: str) -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Brain Tumor Detection Center", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Patient MRI Scan Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Analysis Summary:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Prediction       : {predicted_class.replace('_', ' ').title()}", ln=True)
    pdf.cell(0, 10, f"Confidence       : {confidence:.2f}%", ln=True)
    pdf.cell(0, 10, "Recommendation :", ln=True)

    if predicted_class == 'no_tumor':
        pdf.set_text_color(0, 150, 0)
        pdf.multi_cell(0, 10, "No tumor detected. Regular health checkups are advised. Stay healthy!")
    else:
        pdf.set_text_color(220, 20, 60)
        pdf.multi_cell(0, 10, "Tumor detected. Immediate consultation with a neurologist is strongly recommended.")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Generated automatically by AI Diagnostic System", ln=True, align='C')

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# --- Streamlit App ---

# --- Streamlit App ---

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection        (Using Deep Learing through MRI Scans)")
st.write("Upload MRI images to detect brain tumors and generate a formal PDF report.")

uploaded_files = st.file_uploader("Upload MRI image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

            with st.spinner("🤪 Analyzing... Please wait"):
                predicted_class, confidence, img_array = predict_image(img, model)
                heatmap = make_gradcam_heatmap(img_array, model)
                cam_result = display_gradcam(img, heatmap)

            # Display Prediction
            st.subheader(f"🔍 Prediction: {predicted_class.replace('_', ' ').title()}")
            st.metric(label="Model Confidence", value=f"{confidence*100:.2f}%")
            st.progress(int(confidence * 100))

            st.image(cam_result, caption="Grad-CAM Visualization", use_container_width=True)

            # Appointment Logic
            if predicted_class != "no_tumor":
                st.warning("⚠️ Tumor detected! Please consult a doctor.")
                if st.button(f"📅 Book Doctor Appointment for {uploaded_file.name}"):
                    now = datetime.utcnow()
                    calendar_link = generate_google_calendar_link(
                        appointment_title="Brain Tumor Consultation",
                        details="Consult a neurologist immediately.",
                        start_time=now + timedelta(days=1)
                    )
                    st.markdown(f"[📅 Book Doctor Appointment]({calendar_link})")
            else:
                st.success("✅ No tumor detected. Stay healthy!")

            # Generate PDF Report
            pdf_report = generate_pdf_report(predicted_class, confidence*100, uploaded_file.name)
            st.download_button(
                label="📄 Download Hospital Report (PDF)",
                data=pdf_report,
                file_name=f"{uploaded_file.name.split('.')[0]}_report.pdf",
                mime="application/pdf"
            )

            st.markdown("---")

        except UnidentifiedImageError:
            st.error(f"❌ Error: Unable to process file '{uploaded_file.name}'. Please upload a valid image.")

else:
    st.info("ℹ️ Please upload MRI image(s) to proceed.")

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('coffee_quality_model.h5')
    return model

model = load_model()

CLASS_LABELS = ['Coffee Berry Borer', 'Coffee Damage Bean', 'Coffee Healthy Berry']
CLASS_INFO = {
    'Coffee Healthy Berry': {
        "description": "Biji kopi dalam kondisi sehat tanpa tanda-tanda penyakit atau kerusakan.",
        "remedy": "Tidak diperlukan tindakan perbaikan, biji kopi ini dalam kondisi optimal."
    },
    'Coffee Berry Borer': {
        "description": "Biji kopi ini diserang oleh penggerek buah, hama yang membuat lubang pada buah kopi. Akibatnya, kualitas biji kopi menurun.",
        "remedy": "Gunakan perangkap feromon untuk mengurangi populasi penggerek. Semprotkan insektisida seperti Beauveria bassiana untuk pengendalian hayati."
    },
    'Coffee Damage Bean': {
        "description": "Biji kopi ini menunjukkan tanda-tanda kerusakan fisik, yang dapat disebabkan oleh faktor lingkungan seperti hujan berlebihan atau penanganan yang tidak tepat selama proses panen dan pasca panen.",
        "remedy": "Perbaiki proses panen dan pengolahan pascapanen. Hindari pengeringan biji kopi di bawah sinar matahari langsung yang terlalu kuat."
    }
}

def preprocess_image(image):
    width, height = image.size
    left = int(0.25 * width)
    right = int(0.75 * width)
    top = int(0.25 * height)
    bottom = int(0.75 * height)
    image = image.crop((left, top, right, bottom))

    image = image.resize((240, 240))  
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

with st.sidebar:
        st.image('./testing/KN004_Coffee_Cherry.jpg')
        st.subheader("Deteksi akurat penyakit yang ada pada buah kopi. Hal ini membantu pengguna untuk dengan mudah mendeteksi penyakit dan mengidentifikasi penyebabnya.")

# Page Title
st.title("Coffee Berry Diseases Classification")
st.write("Unggah gambar atau ambil gambar menggunakan kamera untuk mendeteksi penyakit biji kopi.")

# Area untuk upload file
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

# Area untuk mengambil gambar dari kamera
camera_active = st.button("Take a Photo ")

camera_file = None
if camera_active:
    camera_file = st.camera_input("Ambil gambar dari kamera")

# Proses jika ada file yang diunggah atau diambil dari kamera
if uploaded_file is not None or camera_file is not None:
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", width=300)
    elif camera_file:
        image = Image.open(camera_file)
        st.image(image, caption="Gambar dari Kamera", width=300)

    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.sidebar.success(f"**Accuracy :** {confidence:.2f}%")
    string = "Detected Disease : " + CLASS_LABELS[np.argmax(predictions)]
    if CLASS_LABELS[np.argmax(predictions)] == 'Coffee Healthy Berry':
        st.sidebar.success(string)

    elif CLASS_LABELS[np.argmax(predictions)] == 'Coffee Damage Bean':
        st.sidebar.error(string)

    elif CLASS_LABELS[np.argmax(predictions)] == 'Coffee Berry Borer':
        st.sidebar.warning(string)

    # Display Information and Remedy
    st.write(f"## Hasil Prediksi: {predicted_class}")
    st.info(CLASS_INFO[predicted_class]["description"])
    st.markdown("## Rekomendasi:")
    st.warning(CLASS_INFO[predicted_class]["remedy"])


import streamlit as st
from PIL import Image
from modelo_yolov8 import procesar_imagen  # Asumiendo que tienes una función para procesar imágenes con YOLOv8

# Título de la aplicación
st.title('Detector de Objetos YOLOv8')

# Carga de imagen
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

    # Procesamiento de la imagen
    if st.button('Detectar Objetos'):
        # Aquí procesas la imagen con YOLOv8 y obtienes los resultados
        resultados = procesar_imagen(image)
        
        # Mostrar resultados (esto dependerá de cómo tu función devuelva los resultados)
        for resultado in resultados:
            st.write(resultado)

        # Opcionalmente, puedes mostrar la imagen con las cajas delimitadoras dibujadas
        st.image(resultados, caption='Detecciones', use_column_width=True)
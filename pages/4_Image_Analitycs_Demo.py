# Este sería tu archivo app.py para la aplicación Streamlit
import streamlit as st
from PIL import Image
from modelo_yolov8 import cargar_modelo_yolov8, procesar_imagen
import cv2
import numpy as np

# Título de la aplicación
st.title('Detector de Basuras YOLOv8')

# Carga de imagen
uploaded_file = st.file_uploader("Elige una imagen para analizar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

    # Procesamiento de la imagen
    if st.button('Detectar Basura'):
        # Convertir la imagen PIL a un array compatible con OpenCV
        image_array = np.array(image)

        # Ruta al modelo YOLOv8 preentrenado
        ruta_modelo = '/workspaces/ecovision/modelo_yolov8/model/best.pt'

        # Cargar el modelo
        modelo = cargar_modelo_yolov8(ruta_modelo)

        # Realizar la detección
        resultados = procesar_imagen(modelo, image_array)
        imagen_orig = resultados.orig_img

        # Iterar sobre cada caja delimitadora
        for caja in resultados.boxes.xyxy[0]:
            # Extraer las coordenadas de la caja
            x_min, y_min, x_max, y_max, confianza, clase = caja

            # Dibujar la caja delimitadora en la imagen
            cv2.rectangle(imagen_orig, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 255, 0), thickness=2)

            # Obtener el nombre de la clase
            nombre_clase = resultados.names[int(clase)]

            # Mostrar el nombre de la clase y la confianza
            etiqueta = f'{nombre_clase}: {confianza:.2f}'
            cv2.putText(imagen_orig, etiqueta, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar la imagen con las detecciones
        cv2.imshow('Detecciones', imagen_orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                    
        # Convertir de vuelta a PIL Image y mostrar
        st.image(imagen_orig, caption='Detecciones', use_column_width=True)
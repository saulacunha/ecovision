import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import io, json, os
from roboflow import Roboflow

# Inicializar session_state
if 'edit_labels' not in st.session_state:
    st.session_state.edit_labels = False
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

def redibujar_imagen_con_etiquetas(image, data, etiquetas):
    """
    Redibuja la imagen con las nuevas etiquetas.

    Args:
    - image: Imagen original.
    - data: Datos de detección.
    - etiquetas: Lista de etiquetas actualizadas.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, item in enumerate(data):
        box = item['box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
        etiqueta = etiquetas[i]
        draw.text((x1, y1), etiqueta, fill="blue", font=font)

    return image

def crear_archivo_de_anotaciones(data, etiquetas_actualizadas):
    anotaciones = []
    for item, etiqueta in zip(data, etiquetas_actualizadas):
        box = item['box']
        anotacion = {
            "label": etiqueta,
            "coordinates": {
                "x": int((box['x1'] + box['x2']) / 2),
                "y": int((box['y1'] + box['y2']) / 2),
                "width": int(box['x2'] - box['x1']),
                "height": int(box['y2'] - box['y1'])
            }
        }
        anotaciones.append(anotacion)

    with open("annotations.json", "w") as file:
        json.dump(anotaciones, file)

    return "annotations.json"

# Lista de clases para etiquetar
clases = [
    "Aerosol", "Aluminium blister pack", "Aluminium foil", "Battery",
    "Broken glass", "Carded blister pack", "Cigarette", "Clear plastic bottle",
    "Corrugated carton", "Crisp packet", "Disposable food container",
    "Disposable plastic cup", "Drink can", "Drink carton", "Egg carton",
    "Foam cup", "Foam food container", "Food Can", "Food waste", "Garbage bag",
    "Glass bottle", "Glass cup", "Glass jar", "Magazine paper", "Meal carton",
    "Metal bottle cap", "Metal lid", "Normal paper", "Other carton",
    "Other plastic bottle", "Other plastic container", "Other plastic cup",
    "Other plastic wrapper", "Other plastic", "Paper bag", "Paper cup",
    "Paper straw", "Pizza box", "Plastic bottle cap", "Plastic film",
    "Plastic glooves", "Plastic lid", "Plastic straw", "Plastic utensils",
    "Polypropylene bag", "Pop tab", "Rope - strings", "Scrap metal", "Shoe",
    "Single-use carrier bag", "Six pack rings", "Spread tub", "Squeezable tube",
    "Styrofoam piece", "Tissues", "Toilet tube", "Tupperware", "Unlabeled litter",
    "Wrapping paper"
]

# Inicialización de Roboflow
rf = Roboflow(api_key="e1Fg1On5uW2Nm61qupO5")
workspaceId = 'ecovision-cdwmz'
projectId = 'ecovision-cwxzm'
project = rf.workspace(workspaceId).project(projectId)

# Título de la aplicación
st.title('Detector de Basuras YOLOv8')

# Carga de imagen
uploaded_file = st.file_uploader("Elige una imagen para analizar", type=["jpg", "png", "jpeg"])

# Configuraciones de detección
confidence = st.slider("Confianza", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
iou = st.slider("IoU (Intersección sobre Unión)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

    if st.button('Detectar Basura'):
        # Procesar la imagen con la API
        url = "https://api.ultralytics.com/v1/predict/jEeX28mAwk6TlAitYZXG"
        headers = {"x-api-key": "2c218ac4848cb5221ab68bec4f8c7399b92a6ffd1e"}
        data = {"size": 640, "confidence": confidence, "iou": iou}

        # Convertir la imagen a un formato adecuado para el envío
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)

        # Enviar la solicitud
        response = requests.post(url, headers=headers, data=data, files={"image": buffered})

        # Verificar la respuesta
        try:
            response.raise_for_status()
            st.session_state.resultados = response.json()

            if st.session_state.resultados['success']:
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()

                # Dibujar las detecciones iniciales en la imagen
                for item in st.session_state.resultados['data']:
                    box = item['box']
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    label = f"{item['name']} {item['confidence']:.2f}"
                    draw.text((x1, y1), label, fill="red", font=font)

                st.image(image, caption='Detecciones Iniciales', use_column_width=True)
                st.session_state.edit_labels = True
        except requests.HTTPError as e:
            st.write(f"Error en la solicitud: {e}")

    if st.session_state.edit_labels and st.session_state.resultados:
        etiquetas_actualizadas = []
        for i, item in enumerate(st.session_state.resultados['data']):
            etiqueta_inicial = item['name']
            index_etiqueta_inicial = clases.index(etiqueta_inicial) if etiqueta_inicial in clases else 0
            nueva_etiqueta = st.selectbox(f"Objeto {i+1}: {etiqueta_inicial}", clases, index=index_etiqueta_inicial, key=f"objeto_{i}_edit")
            etiquetas_actualizadas.append(nueva_etiqueta)

        if st.button("Guardar Etiquetas"):
            # Actualizar la imagen con las etiquetas actualizadas
            image = redibujar_imagen_con_etiquetas(image, st.session_state.resultados['data'], etiquetas_actualizadas)
            st.image(image, caption='Imagen con Etiquetas Actualizadas', use_column_width=True)

        # Preguntar si desea guardar los cambios
        if st.checkbox("¿Guardar en Roboflow?"):
            buffered_f = io.BytesIO()
            image.save(buffered_f, format="JPEG")
            img_byte = buffered_f.getvalue()
            with open('img.jpg', 'wb') as f:
                f.write(img_byte)
            annotations = crear_archivo_de_anotaciones(st.session_state.resultados['data'], etiquetas_actualizadas)
            project.upload(image_path='img.jpg', annotations=annotations)
            st.success("Imagen y etiquetas guardadas en Roboflow")
            # os.remove('img.jpg')

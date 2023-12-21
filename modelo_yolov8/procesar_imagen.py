from ultralytics import YOLO

def cargar_modelo_yolov8(ruta_modelo):
    # Cargar el modelo YOLOv8
    modelo = YOLO(ruta_modelo)
    return modelo

def procesar_imagen(modelo, imagen_array):
    # Realizar la predicción
    resultados = modelo.predict(imagen_array)[0]
    return resultados

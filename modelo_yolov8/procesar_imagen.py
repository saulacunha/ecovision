import torch
from PIL import Image
from torchvision.transforms import functional as F

def cargar_modelo_yolov8(ruta_modelo):
    # Carga el modelo YOLOv8 desde la ruta especificada
    modelo = torch.load(ruta_modelo)
    modelo.eval()  # Poner el modelo en modo de evaluación
    return modelo

def procesar_imagen(imagen):
    # Cargar el modelo (idealmente esto se debería hacer una sola vez, no en cada llamada)
    modelo = cargar_modelo_yolov8('ruta/a/tu/modelo/yolov8.pth')

    # Convertir la imagen PIL a tensor
    imagen_tensor = F.to_tensor(imagen).unsqueeze(0)  # Añadir una dimensión de batch

    # Realizar la detección
    with torch.no_grad():
        predicciones = modelo(imagen_tensor)

    # Convertir predicciones a un formato útil
    resultados = []
    for caja, etiqueta, score in zip(predicciones['boxes'], predicciones['labels'], predicciones['scores']):
        caja = caja.numpy().tolist()  # Convertir la caja delimitadora a una lista
        etiqueta = etiqueta.item()   # Obtener el valor de la etiqueta
        score = score.item()         # Obtener el valor del score
        resultados.append({'caja': caja, 'etiqueta': etiqueta, 'score': score})

    return resultados

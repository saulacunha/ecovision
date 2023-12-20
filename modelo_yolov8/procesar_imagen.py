import onnxruntime as ort
from PIL import Image
from torchvision.transforms import functional as F

def cargar_modelo_yolov8(ruta_modelo):
    # Cargar el modelo ONNX
    sesion_onnx = ort.InferenceSession(ruta_modelo)
    return sesion_onnx

def procesar_imagen(imagen, sesion_onnx):
    # Convertir la imagen PIL a tensor y luego a numpy array
    imagen_tensor = F.to_tensor(imagen).unsqueeze(0).numpy()

    # Obtener los nombres de las entradas y salidas del modelo ONNX
    entradas = [input.name for input in sesion_onnx.get_inputs()]
    salidas = [output.name for output in sesion_onnx.get_outputs()]

    # Realizar la detección
    predicciones = sesion_onnx.run(salidas, {entradas[0]: imagen_tensor})

    # Aquí necesitarás ajustar la manera en que procesas 'predicciones'
    # dependiendo de cómo YOLOv8 en ONNX estructura sus salidas.

    return predicciones  # Modificar según sea necesario

# Cargar el modelo una vez
sesion_modelo = cargar_modelo_yolov8('/workspaces/ecovision/modelo_yolov8/model/best.onnx')

# Ejemplo de cómo usar la función 'procesar_imagen'
# imagen = Image.open('/ruta/a/una/imagen.jpg')
# resultados = procesar_imagen(imagen, sesion_modelo)


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
    modelo = cargar_modelo_yolov8('/workspaces/ecovision/modelo_yolov8/model/best.onnx')

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

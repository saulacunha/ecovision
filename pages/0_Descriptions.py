# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code


# Título de la página
st.title("EcoVision: Detección y Análisis de Residuos con IA")

# Introducción y Objetivos
st.header("Introducción")
st.write("""
En el contexto contemporáneo de preocupaciones ambientales y avances tecnológicos, 
         la gestión de residuos, particularmente los plásticos, 
         ha emergido como una temática crítica en el ámbito de las ciencias ambientales y la sostenibilidad. 
         A medida que el mundo produce y consume volúmenes cada vez mayores de plásticos, 
         las playas globales han sufrido las consecuencias, convirtiéndose en puntos críticos de acumulación de estos desechos. Este escenario ha propiciado la 
         necesidad de una investigación interdisciplinaria que combine tecnología, como la Inteligencia Artificial (IA), 
         con enfoques ambientales para abordar eficientemente el problema. Las técnicas tradicionales de monitorización, 
         que a menudo dependen de la observación manual y muestreos físicos, han demostrado ser insuficientes en escala y precisión, 
         impulsando la búsqueda de soluciones tecnológicas innovadoras.
         """)
st.write("""
La contaminación por residuos plásticos ha dejado una huella indeleble en el ecosistema marino. 
         Sin embargo, el impacto de esta contaminación no se limita a los océanos y su biodiversidad; 
         tiene repercusiones directas en la salud humana, en la sostenibilidad ambiental y en la economía energética.
         """)
st.write("""
Desde un punto de vista de salud humana [1],los fragmentos microplásticos en playas y océanos pueden ser ingeridos por la fauna marina, 
         introduciendo toxinas en la cadena alimentaria que, eventualmente, 
         llegan a los seres humanos a través del consumo de productos del mar. 
         Estos contaminantes tienen el potencial de causar problemas de salud, desde trastornos digestivos hasta endocrinológicos.
         """)
st.write("""
Desde una perspectiva ambiental [2], [3], más allá del daño visible a nuestros paisajes costeros, 
         la acumulación de plásticos amenaza la biodiversidad marina, causando la muerte de innumerables organismos, 
         desde microorganismos hasta grandes mamíferos marinos. El equilibrio de los ecosistemas costeros y marinos se ve gravemente afectado, con repercusiones que pueden perdurar durante generaciones.
En cuanto al ahorro energético [4], [5], el uso de técnicas basadas en IA para la monitorización y gestión de residuos reduce significativamente la necesidad de operaciones manuales y muestreos extensivos, 
         tradicionalmente energéticamente costosos. Al optimizar la detección y gestión de áreas contaminadas, se pueden planificar intervenciones más eficientes, reduciendo el gasto energético global del proceso.
         """)
st.header("Objetivos")
st.subheader('Generales')
st.write("""
•	Desarrollar un sistema basado en inteligencia artificial que permita detectar residuos en imágenes de entornos costeros introducidas por el usuario. \n
•	Proporcionar una interfaz intuitiva que facilite la introducción de imágenes por parte del usuario y visualice de manera efectiva los resultados de la detección.
         """)
st.subheader('Específicos')
st.write("""
•	Tratar y preparar un conjunto de datos de entornos costeros que sea adecuado para el entrenamiento y validación del modelo de IA. \n
•	Diseñar y entrenar un modelo de inteligencia artificial capaz de detectar de manera precisa los residuos en las imágenes. \n
•	Implementar un sistema de análisis que evalúe la eficacia del modelo, utilizando métricas pertinentes y proporcionando insights claros sobre su rendimiento. \n
•	Desarrollar una interfaz de usuario que permita la carga de imágenes, muestre los resultados de la detección y proporcione una retroalimentación visual clara. 
         """)
st.header("Metodología y Tecnología")
st.write("""
         En EcoVision, hemos adoptado una metodología centrada en el uso de YOLOv8 para entrenar nuestro modelo de inteligencia artificial. Este modelo fue entrenado con un conjunto diverso de 3597 imágenes, abarcando 59 categorías de residuos, a lo largo de 100 epochs. Nuestro enfoque se ha enfocado en lograr una identificación rápida y precisa de los desechos, lo que es crucial para la gestión efectiva de residuos en entornos costeros.
         """)
st.image('https://docs.google.com/drawings/d/e/2PACX-1vTTIOyMWM7_fhmrJ5BnMoS_ohLCcxIHXf5k3Prl3lb_HUpIKoIeaS8yGsE6yXTohLvnCTeFEDSxgwxo/pub?w=1440&h=1080', caption='Distribución de Dataset')
st.image('https://docs.google.com/drawings/d/e/2PACX-1vRqAUC7hwRS6EcUyLWquSbBx8-nqXM7Jj8eleGI04oPL1rvsg6aZOJjjq7XFZBsQpXgoYoxz5tfHSJa/pub?w=1440&h=1080', caption='Ejemplo de Dataset')
st.image('https://docs.google.com/drawings/d/e/2PACX-1vTOBjLSStFb0VvUW83Rtx-HokgZcWTV7tj8AUxYA89CDGY70nAJnho6oOA-pnCT01-dhJTjzrDRQmH7/pub?w=1441&h=685', caption='Modelo Ecovision Ultralytics')
# # Resultados
# st.header("Resultados")
# st.write("""
#          Presentación de los resultados clave del proyecto, como la eficacia y precisión del modelo de IA en la detección de residuos.
#          """)

# Conclusiones y Futuro
st.header("Conclusiones y Futuro")
st.write("""
         Las conclusiones de este estudio reflejan un análisis exhaustivo de los resultados obtenidos tras la implementación y evaluación del modelo de inteligencia artificial YOLOv8 para la detección de residuos en entornos costeros. Los resultados han confirmado la hipótesis inicial de que un modelo de aprendizaje profundo puede efectivamente identificar y clasificar residuos con una alta tasa de precisión en imágenes de diversas condiciones ambientales.
         De forma más específica, el modelo demostró una capacidad notable para generalizar a partir de los datos de entrenamiento a situaciones del mundo real, como se evidencia en los altos valores de mAP alcanzados durante la fase de validación. 
""")
st.write("""
         Esta eficacia estaba en línea con las expectativas, aunque el grado de precisión en las clasificaciones bajo ciertas condiciones de iluminación y ángulos de imagen fue sorprendentemente positivo, superando las estimaciones iniciales.
""")
st.write(""" No obstante, hubo aspectos que no se alinearon completamente con las proyecciones iniciales. 
         Por ejemplo, la clasificación de ciertos tipos de residuos con apariencias similares presentó desafíos mayores de los anticipados, lo que sugiere la necesidad de un entrenamiento adicional del modelo o la revisión de las clases de residuos en el conjunto de datos.
En términos de rendimiento, la velocidad de procesamiento cumplió con los objetivos propuestos, ofreciendo detecciones en tiempo real que son esenciales para aplicaciones prácticas como la monitorización de entornos naturales y la gestión de residuos.
La investigación también reveló áreas de mejora en el preprocesamiento de datos y la necesidad de un aumento de datos más robusto para mejorar la generalización del modelo. 
""")
st.write("""  Estos hallazgos ponen de manifiesto la importancia de una preparación de datos meticulosa para la eficacia del aprendizaje automático y la visión por computadora.
En resumen, los resultados respaldan la viabilidad de utilizar YOLOv8 en la detección y clasificación de residuos, aunque también indican oportunidades claras para la optimización y el avance en la precisión del modelo. Estas conclusiones proporcionan una base sólida para trabajos futuros y aplicaciones prácticas en la conservación ambiental y la sostenibilidad
 """)

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

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Ecovision",
        page_icon="👋",
    )

    st.write("# Bienvenido a Ecovision! 👋")
    # Título principal y contexto del proyecto
    st.header('Demostración de Detección de Objetos con YOLOv8')
    # Descripción detallada del proyecto y su propósito
    st.write('''
    ECOVISION es el resultado de un ambicioso proyecto de Trabajo de Fin de Máster (TFM) que explora las fronteras de la inteligencia artificial aplicada a la sostenibilidad ambiental. Este sistema interactivo demuestra el poder de YOLOv8, la última innovación en detección de objetos, para identificar y analizar elementos en entornos costeros con una precisión y velocidad sin precedentes.
    ''')

    # Características clave del proyecto ECOVISION
    st.subheader('Características Clave del Proyecto ECOVISION:')
    st.write('''
    - **Avanzado y Preciso**: Incorporando YOLOv8, ECOVISION detecta objetos con una precisión y velocidad revolucionarias.
    - **Interfaz Intuitiva para Demostración**: La demostración de ECOVISION se presenta a través de una interfaz intuitiva desarrollada con Streamlit, diseñada para facilitar la interacción de los usuarios.
    - **Aplicaciones en Tiempo Real**: ECOVISION es capaz de procesar imágenes y video en tiempo real, ilustrando las posibilidades de la tecnología en aplicaciones prácticas.
    ''')

    # Sección de aplicaciones prácticas de ECOVISION
    st.subheader('Aplicaciones Prácticas de ECOVISION:')
    st.write('''
    Como parte de un proyecto académico de vanguardia, ECOVISION tiene aplicaciones en el monitoreo ambiental, la gestión de recursos naturales y mucho más, demostrando la utilidad práctica de la inteligencia artificial en la sostenibilidad ambiental.
    ''')

    # Instrucciones para la demostración
    st.subheader('Demostración Interactiva:')
    st.write('''
    Explora la capacidad de ECOVISION cargando tus propias imágenes y observa cómo el sistema identifica y clasifica objetos en tiempo real. Esta demostración es un escaparate del trabajo realizado y de las tecnologías implementadas durante el TFM.
    ''')

if __name__ == "__main__":
    run()

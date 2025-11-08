### 7) 🖼️ Buscador visual por similitud (10)

# **Objetivo:** Dada una imagen, encontrar similares por embeddings o etiquetas.

# **Mínimos**
# - Dataset local (assets/).
# - Indexación por embeddings/tags.
# - Consulta: top-k similares con score.

# **Arquitectura**
# - Preproceso → embeddings (CLIP/Vision) → vector store → similitud → grid UI.

# **APIs/Modelos**
# - Azure: Vision (tags) + (opcional) embeddings con Azure OpenAI (CLIP-like).
# - Alternativas: HF openai/clip-vit-base-patch32; Google Vision labels.

# **Páginas**
# - “Indexar”.
# - “Buscar”.

# **Ampliaciones**
# - Filtro por etiquetas.
# - Búsqueda por texto (“playa al atardecer”).

import requests, os, sys
from dotenv import load_dotenv
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
    
from controller.IndexImageController import IndexImagenesController
from controller.VectorDBController import VectorDBController

# Carga las variables del archivo .env (solo en local)
load_dotenv()

# Configuraciones de API de: 
subscription_key = os.getenv("SUBSCRIPTION_KEY")
endpoint = os.getenv("ENDPOINT")

st.title("Proyecto Final:🖼️ Buscador visual por similitud")

col1, col2 = st.columns(2)



# Request headers.
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

# Request parameters. All of them are optional.
params = {
    'visualFeatures': 'tags',
    'language': 'es',
}



# *- 1 Indexar imagenes de assets/
# Recorrer la carpeta y leer las rutas de las imágenes.
# *- 2 Obtener embeddings o tags
# Usar modelos preentrenados que generan vectores de características visuales, por ejemplo:
# CLIP (de OpenAI, vía open_clip o transformers)
# ResNet50 (desde torchvision.models)
# ViT (Vision Transformer)
# Cada imagen → un vector numérico (embedding).
with st.spinner('🔍 Indexando imágenes, por favor espera...'):
    try:
        indexador = IndexImagenesController(
            assets_dir="assets",
            output_file="image_index.json"
        )
        indexador.crear_index()
    except Exception as e:
        st.error(f"❌ Error al indexar imágenes: {e}")

# *- 3 Almacenar en vector DB (FAISS u otra)
# Utilizar FAISS para indexar los embeddings y permitir búsquedas rápidas por similitud.
# *- *4 Utilizar top-k similitud
# FAISS te permite buscar los k más similares
with st.spinner('📦 Creando índice FAISS...'):
    try:
        vector_db = VectorDBController(
            index_file="image_index.json",
            faiss_index_file="faiss_index.bin"
        )
        vector_db.cargar_embeddings()
        vector_db.crear_faiss_index()
        st.success("✅ Índice FAISS creado con éxito.")
    except Exception as e:
        st.error(f"❌ Error al crear índice FAISS: {e}")


# TODO:
# - 5 Implementar búsqueda por similitud
# Pasar una imagen de consulta, generar su embedding y comparar con FAISS.


# TODO:
# - 6 Mostrar resultados en grid UI
# Usar Streamlit para mostrar las imágenes similares en una cuadrícula.


st.markdown("---")
st.markdown("© 2025 - Proyecto de Programación de Inteligencia Artificial: Buscador visual por similitud")
st.markdown("Desarrollado por [jbotgil] [https://github.com/jbotgil]")
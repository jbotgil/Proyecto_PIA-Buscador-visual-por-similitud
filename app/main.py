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


import requests, os
from dotenv import load_dotenv
import streamlit as st

# Carga las variables del archivo .env (solo en local)
load_dotenv()

# 🔐 Configuración Azure Speech
SPEECH_KEY = os.getenv("SPEECH_KEY")
REGION = os.getenv("REGION")

st.title("🗣️ Síntesis de voz")


st.title("Proyecto Final: Generador de similitudes a traves de una imagen")


col1, col2 = st.columns(2)

# Configuraciones de API de: 
subscription_key = os.getenv("SBSCRIPTION_KEY")
endpoint = os.getenv("ENDPOINT")

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

# # Solicitamos una imagen
# with col1:
#     imagen = st.file_uploader("Introduce una imagen:", type=['png', 'jpg'])
# with col2:
#     if imagen is not None:
#         st.image(imagen, caption="Vista previa", use_container_width=True)
#     else:
#         st.text("No has introducido ninguna imagen")
        
# botonSubmit = st.button("Analizar imagen:")

# # Una vez introducida la imagen
# if botonSubmit:
#     if imagen is not None:
#         image_data = imagen.read()

#         try:
#             response = requests.post(
#                 endpoint,
#                 headers=headers,
#                 params=params,
#                 data=image_data
#             )
#             response.raise_for_status()

#             result = response.json()
#             st.json(result) 
            
#             captions = result['tags']
#             for caption in captions:
#                 st.success(f"{caption['name']} (confianza: {caption['confidence']:.2%})")
            
#         except Exception as ex:
#             st.error(f"Error al analizar la imagen: {ex}")
#     else:
#         st.error("No has introducido ninguna imagen.")
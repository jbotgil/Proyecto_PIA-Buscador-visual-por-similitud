### 7) 🖼️ Buscador visual por similitud (10)

import requests, os, sys
from dotenv import load_dotenv
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
    
from controller.IndexImageController import IndexImagenesController
from controller.VectorDBController import VectorDBController

# Carga las variables del archivo .env (solo en local)
load_dotenv()

st.title("Proyecto Final:🖼️ Buscador visual por similitud")


#* 1️ Cachear y cargar modelo CLIP
@st.cache_resource
def cargar_modelo_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = cargar_modelo_clip()

#* 2️ Cachear generación de embeddings de imagen
@st.cache_data(show_spinner=False)
def generar_embedding_imagen(image_bytes):

    image = Image.open(image_bytes).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_features = model.get_image_features(**inputs)
    query_features = query_features / query_features.norm(p=2)
    return query_features.cpu().numpy().flatten().tolist()

#* 3️ Indexar imágenes con CLIP
with st.spinner('🔍 Indexando imágenes, por favor espera...'):
    try:
        indexador = IndexImagenesController(
            assets_dir="assets",
            output_file="image_index.json"
        )
        indexador.crear_index()
    except Exception as e:
        st.error(f"❌ Error al indexar imágenes: {e}")

#* 4️ Crear índice FAISS
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

#* 5️ Subir imagen de consulta y generar embedding
uploaded_file = st.file_uploader("Sube una imagen para buscar similares", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption="📷 Imagen de consulta", width=300)

    # Generar embedding cacheado
    with st.spinner("Generando embedding de la imagen..."):
        query_embedding = generar_embedding_imagen(uploaded_file)

    # Buscar similares con FAISS
    with st.spinner("Buscando imágenes más parecidas..."):
        resultados = vector_db.buscar_similares(query_embedding, top_k=10)

    #* 6️ Mostrar resultados en grid UI
    if resultados:
        st.subheader("🖼️ Imágenes más similares encontradas:")
        num_cols = min(len(resultados), 10)
        cols = st.columns(num_cols)

        for i, res in enumerate(resultados):
            col = cols[i % num_cols]  # distribuir las imágenes en la cuadrícula
            with col:
                st.image(res["path"], use_container_width=True)
                st.caption(f"🔹 Score: {res['score']:.4f}")

st.markdown("---")
st.markdown("© 2025 - Proyecto de Programación de Inteligencia Artificial: Buscador visual por similitud")
st.markdown("Desarrollado por [jbotgil] [https://github.com/jbotgil]")
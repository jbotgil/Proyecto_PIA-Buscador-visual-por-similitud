import os, sys
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from controller.IndexImageController import IndexImagenesController
from controller.VectorDBController import VectorDBController

st.title("Proyecto Final:🖼️ Buscador visual por similitud")

# 1️⃣ Cachear modelo CLIP
@st.cache_resource
def cargar_modelo_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = cargar_modelo_clip()

# 2️⃣ Cachear embeddings de consulta
@st.cache_data(show_spinner=False)
def generar_embedding_imagen(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_features = model.get_image_features(**inputs)
    query_features = query_features / query_features.norm(p=2)
    return query_features.cpu().numpy().flatten().tolist()


# 3️⃣ Crear índice solo una vez por sesión
if "index_creado" not in st.session_state:
    index_file = "image_index.json"
    if not os.path.exists(index_file):
        with st.spinner('🔍 Indexando imágenes, por favor espera...'):
            indexador = IndexImagenesController(
                assets_dir="assets",
                output_file=index_file
            )
            indexador.crear_index()
        st.success("✅ Índice de imágenes creado con éxito (primera ejecución).")
    else:
        st.info("ℹ️ Índice ya existente. No se vuelve a crear automáticamente.")
    st.session_state.index_creado = True


# 4️⃣ Botón manual para reindexar si agregas nuevas fotos
if st.button("🔄 Reindexar imágenes manualmente"):
    with st.spinner("Reindexando todas las imágenes..."):
        indexador = IndexImagenesController(
            assets_dir="assets",
            output_file="image_index.json"
        )
        indexador.crear_index()
    st.success("✅ Reindexación completa.")
    # Actualiza el estado para evitar que vuelva a hacerlo
    st.session_state.index_creado = True


# 5️⃣ Cachear FAISS
@st.cache_resource
def cargar_vector_db():
    vector_db = VectorDBController(
        index_file="image_index.json",
        faiss_index_file="faiss_index.bin"
    )
    vector_db.cargar_embeddings()
    vector_db.crear_faiss_index()
    return vector_db

vector_db = cargar_vector_db()


# 6️⃣ Subir imagen y buscar similares
uploaded_file = st.file_uploader("Sube una imagen para buscar similares", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption="📷 Imagen de consulta", width=300)

    with st.spinner("Generando embedding de la imagen..."):
        query_embedding = generar_embedding_imagen(uploaded_file)

    with st.spinner("Buscando imágenes más parecidas..."):
        resultados = vector_db.buscar_similares(query_embedding, top_k=10)

    if resultados:
        st.subheader("🖼️ Imágenes más similares encontradas:")
        num_cols = min(len(resultados), 10)
        cols = st.columns(num_cols)
        for i, res in enumerate(resultados):
            with cols[i % num_cols]:
                st.image(res["path"], use_container_width=True)
                st.caption(f"🔹 Score: {res['score']:.4f}")

st.markdown("---")
st.markdown("© 2025 - Proyecto de Programación de Inteligencia Artificial: Buscador visual por similitud")
st.markdown("Desarrollado por [jbotgil] [https://github.com/jbotgil]")
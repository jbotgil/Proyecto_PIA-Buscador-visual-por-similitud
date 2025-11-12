from pathlib import Path
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sys
import tempfile

# Base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent  # /app

# Asegurarnos de que 'controller' se pueda importar
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from controller.IndexImageController import IndexImagenesController
from controller.VectorDBController import VectorDBController

st.title("Proyecto Final:🖼️ Buscador visual por similitud")

# 1️⃣ Cargar modelo CLIP
@st.cache_resource
def cargar_modelo_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = cargar_modelo_clip()

# 2️⃣ Generar embedding de imagen
@st.cache_data(show_spinner=False)
def generar_embedding_imagen(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(p=2)
    return features.cpu().numpy().flatten().tolist()

# 3️⃣ Crear índice si no existe
index_file = BASE_DIR / "image_index.json"
assets_dir = BASE_DIR / "assets"

if "index_creado" not in st.session_state:
    if not index_file.exists():
        with st.spinner("🔍 Indexando imágenes..."):
            indexador = IndexImagenesController(
                assets_dir=assets_dir,
                output_file=index_file
            )
            indexador.crear_index()
        st.success("✅ Índice creado.")
    st.session_state.index_creado = True

# 4️⃣ Reindexación manual
if st.button("🔄 Reindexar imágenes manualmente"):
    with st.spinner("Reindexando imágenes..."):
        indexador = IndexImagenesController(
            assets_dir=assets_dir,
            output_file=index_file
        )
        indexador.crear_index()
    st.success("✅ Reindexación completa.")
    st.session_state.index_creado = True

# 5️⃣ Cargar FAISS
@st.cache_resource
def cargar_vector_db():
    vector_db = VectorDBController(
        index_file=str(index_file),
        faiss_index_file=str(BASE_DIR / "faiss_index.bin")
    )
    vector_db.cargar_embeddings()
    vector_db.crear_faiss_index()
    return vector_db

vector_db = cargar_vector_db()

# 6️⃣ Subir imagen y buscar similares
uploaded_file = st.file_uploader("Sube una imagen para buscar similares", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Crear archivo temporal en Linux
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)

    st.image(tmp_path, caption="📷 Imagen subida", use_container_width=True)

    # Generar embedding y buscar similares
    query_embedding = generar_embedding_imagen(tmp_path)
    resultados = vector_db.buscar_similares(query_embedding, top_k=10)

    if resultados:
        st.subheader("🖼️ Imágenes más similares encontradas:")
        num_cols = min(len(resultados), 10)
        cols = st.columns(num_cols)
        for i, res in enumerate(resultados):
            img_path = Path(res["path"])
            if img_path.exists():
                with cols[i % num_cols]:
                    st.image(img_path, use_container_width=True)
                    st.caption(f"🔹 Score: {res['score']:.4f}")
            else:
                st.warning(f"No se encontró la imagen: {img_path}")

st.markdown("---")
st.markdown("© 2025 - Proyecto de IA: Buscador visual por similitud")
st.markdown("Desarrollado por [jbotgil](https://github.com/jbotgil)")
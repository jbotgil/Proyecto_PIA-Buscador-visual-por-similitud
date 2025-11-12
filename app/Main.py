from pathlib import Path
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Carpeta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel desde 'app'

# Asegurarnos de que 'controller' se pueda importar
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from controller.IndexImageController import IndexImagenesController
from controller.VectorDBController import VectorDBController

st.title("Proyecto Final:🖼️ Buscador visual por similitud")

# 1️⃣ Cachear modelo CLIP
@st.cache_resource
def cargar_modelo_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", force_download=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", force_download=True)
    return model, processor, device

model, processor, device = cargar_modelo_clip()

# 2️⃣ Cachear embeddings de consulta
@st.cache_data(show_spinner=False)
def generar_embedding_imagen(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_features = model.get_image_features(**inputs)
    query_features = query_features / query_features.norm(p=2)
    return query_features.cpu().numpy().flatten().tolist()

# 3️⃣ Crear índice si no existe
if "index_creado" not in st.session_state:
    index_file = BASE_DIR / "image_index.json"
    assets_dir = BASE_DIR / "assets"

    if not index_file.exists():
        with st.spinner('🔍 Indexando imágenes, por favor espera...'):
            indexador = IndexImagenesController(
                assets_dir=assets_dir,
                output_file=index_file
            )
            indexador.crear_index()
        st.success("✅ Índice creado.")
    else:
        st.info("ℹ️ Índice ya existente.")
    st.session_state.index_creado = True

# 4️⃣ Reindexación manual
if st.button("🔄 Reindexar imágenes manualmente"):
    with st.spinner("Reindexando..."):
        indexador = IndexImagenesController(
            assets_dir=BASE_DIR / "assets",
            output_file=BASE_DIR / "image_index.json"
        )
        indexador.crear_index()
    st.success("✅ Reindexación completa.")
    st.session_state.index_creado = True

# 5️⃣ Cachear FAISS
@st.cache_resource
def cargar_vector_db():
    vector_db = VectorDBController(
        index_file=str(BASE_DIR / "image_index.json"),
        faiss_index_file=str(BASE_DIR / "faiss_index.bin")
    )
    vector_db.cargar_embeddings()
    vector_db.crear_faiss_index()
    return vector_db

vector_db = cargar_vector_db()

# 6️⃣ Subir imagen y buscar similares
uploaded_file = st.file_uploader("Sube una imagen para buscar similares", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    save_dir = BASE_DIR / "assets"
    save_dir.mkdir(exist_ok=True)

    save_path = (save_dir / uploaded_file.name).as_posix()  # ✅ Forzar ruta Linux
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(save_path, caption="📷 Imagen subida", width=300)

    query_embedding = generar_embedding_imagen(Path(save_path))
    resultados = vector_db.buscar_similares(query_embedding, top_k=10)

    if resultados:
        st.subheader("🖼️ Imágenes más similares encontradas:")
        num_cols = min(len(resultados), 10)
        cols = st.columns(num_cols)
        for i, res in enumerate(resultados):
            img_path = Path(res["path"]).as_posix()  # ✅ Forzar ruta Linux
            if Path(img_path).exists():
                with cols[i % num_cols]:
                    st.image(img_path, use_container_width=True)
                    st.caption(f"🔹 Score: {res['score']:.4f}")
            else:
                st.warning(f"No se encontró la imagen: {img_path}")

st.markdown("---")
st.markdown("© 2025 - Proyecto de IA: Buscador visual por similitud")
st.markdown("Desarrollado por [jbotgil](https://github.com/jbotgil)")
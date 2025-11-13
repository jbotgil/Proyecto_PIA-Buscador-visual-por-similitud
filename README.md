# 🖼️ Buscador Visual por Similitud (CLIP + FAISS + Streamlit)

Proyecto que permite buscar imágenes por similitud visual usando **CLIP** (Hugging Face) para generar embeddings y **FAISS** para búsquedas vectoriales, con una interfaz sencilla en **Streamlit**.

---

## 📖 Descripción
Sube una imagen y el sistema devuelve las imágenes más parecidas de la carpeta `assets/`.
Añade las imagenes que necesites en `assets/` para tener tu propio buscador de imagenes por similitud local.
Se indexan las imágenes con CLIP (embeddings) y se construye un índice FAISS para búsquedas rápidas.

---

## 📁 Estructura del proyecto

```
app/
├── Main.py                         # Interfaz principal (Streamlit)
├── assets/                         # Carpeta con imágenes a indexar
│   ├── imagen1.jpg
│   └── ...
├── controller/
│   ├── IndexImageController.py     # Genera embeddings y guarda image_index.json
│   └── VectorDBController.py       # Carga embeddings y crea/usa índice FAISS
├── image_index.json                # Generado: lista de {path, embedding}
└── faiss_index.bin                 # Generado: índice FAISS binario
```

---

## 🧩 Requisitos (requirements.txt)

```
streamlit
torch
transformers
Pillow
faiss-cpu
tqdm
numpy
```

> Si tienes GPU y CUDA, instala la versión de `torch` compatible con tu CUDA en vez de la que pip instala por defecto.

---

## ⚙️ Instalación rápida

```bash
git clone https://github.com/jbotgil/Proyecto_PIA-Buscador-visual-por-similitud.git
cd visual-search-clip/

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Ejecutar la aplicación

```bash
streamlit run app/Main.py
```

- En la primera ejecución, si no existen `image_index.json` ni `faiss_index.bin`, la app indexará automáticamente las imágenes en `assets/`.
- Sube una imagen vía el uploader para buscar las más similares (se muestran hasta 10).

---

## 🐳 Imagen Docker disponible

Puedes encontrar una imagen Docker preformada del proyecto en:  
👉 [https://hub.docker.com/repository/docker/jbotgil/proyecto-pia-buscador-similitud/general](https://hub.docker.com/repository/docker/jbotgil/proyecto-pia-buscador-similitud/general)

---


### 🚀 Cómo ejecutar el contenedor

Para lanzar la aplicación desde Docker, ejecuta el siguiente comando:

```bash
docker run -d -p 8501:8501 jbotgil/proyecto-pia-buscador-similitud


## 🪪 Licencia
MIT — libre para uso y modificación.

---

## 👨‍💻 Autor
**Javier Botella Gil** — estudiante y desarrollador.  
GitHub: [@jbotgil](https://github.com/jbotgil)

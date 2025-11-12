from transformers import CLIPProcessor, CLIPModel  # Modelo CLIP de Hugging Face
from PIL import Image                              # Librería Pillow para abrir imágenes
import torch                                       # PyTorch para ejecutar el modelo CLIP
import json                                        # Para guardar y cargar embeddings
from tqdm import tqdm                              # Barra de progreso
from pathlib import Path                           # ✅ Manejo multiplataforma de rutas


class IndexImagenesController:
    def __init__(self, assets_dir="assets", output_file="image_index.json"):
        # ✅ Convertimos las rutas a Path, que funciona igual en Linux o Windows
        self.assets_dir = Path(assets_dir)
        self.output_file = Path(output_file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Carga modelo CLIP preentrenado (base)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def crear_index(self):
        image_data = []

        # Iterar por imágenes válidas
        for img_path in tqdm(self.assets_dir.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image = Image.open(img_path).convert("RGB")

                # Generar embedding CLIP
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                # Normalizar el vector
                image_features = image_features / image_features.norm(p=2)
                embedding = image_features.cpu().numpy().flatten().tolist()

                # ✅ Guardar siempre la ruta en formato POSIX (con /)
                image_data.append({
                    "path": img_path.as_posix(),
                    "embedding": embedding
                })

        # Guardar embeddings
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(image_data, f, indent=2)
        print(f"✅ Index creado con {len(image_data)} imágenes.")
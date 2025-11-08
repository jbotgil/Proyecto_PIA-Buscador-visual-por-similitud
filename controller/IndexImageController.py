from transformers import CLIPProcessor, CLIPModel  # Modelo CLIP de Hugging Face: genera embeddings de imágenes y texto
from PIL import Image                              # Librería Pillow para abrir y procesar imágenes
import torch                                       # PyTorch: ejecuta el modelo CLIP y operaciones con tensores
import os, json                                    # os: manejo de rutas y archivos / json: guardar y cargar embeddings
from tqdm import tqdm                              # Barra de progreso para visualizar el avance al indexar imágenes


class IndexImagenesController:
    def __init__(self, assets_dir="assets", output_file="image_index.json"):
        self.assets_dir = assets_dir
        self.output_file = output_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carga modelo CLIP preentrenado (base)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def crear_index(self):
        image_data = []

        for img_name in tqdm(os.listdir(self.assets_dir)): # Genera barra de prograso
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.assets_dir, img_name)
                image = Image.open(img_path).convert("RGB")

                # Generar embedding CLIP
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs) # Extrae características de la imagen

                # Normalizar el vector
                image_features = image_features / image_features.norm(p=2)
                embedding = image_features.cpu().numpy().flatten().tolist()

                image_data.append({
                    "path": img_path,
                    "embedding": embedding
                })

        # Guardar embeddings
        with open(self.output_file, "w") as f:
            json.dump(image_data, f, indent=2)
        print(f"✅ Index creado con {len(image_data)} imágenes.")

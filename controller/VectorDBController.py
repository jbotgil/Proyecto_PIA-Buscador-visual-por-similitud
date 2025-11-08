import faiss  # Librería para búsquedas vectoriales eficientes
import numpy as np
import json

class VectorDBController:
    def __init__(self, index_file="image_index.json", faiss_index_file="faiss_index.bin"):
        self.index_file = index_file
        self.faiss_index_file = faiss_index_file
        self.embeddings = []
        self.paths = []
        self.index = None

    def cargar_embeddings(self):
        # Carga los embeddings previamente generados por CLIP
        with open(self.index_file, "r") as f:
            data = json.load(f)
        self.paths = [item["path"] for item in data]
        self.embeddings = np.array([item["embedding"] for item in data]).astype("float32")
        print(f"📦 {len(self.embeddings)} embeddings cargados.")
        return self.embeddings

    def crear_faiss_index(self):
        if self.embeddings is None or len(self.embeddings) == 0:
            self.cargar_embeddings()

        d = self.embeddings.shape[1]  # Dimensión del embedding (por ej. 512)
        self.index = faiss.IndexFlatIP(d)  # IP = Inner Product (coseno normalizado)
        faiss.normalize_L2(self.embeddings)  # Normalizar para similitud coseno
        self.index.add(self.embeddings)  # Añadir todos los vectores al índice
        faiss.write_index(self.index, self.faiss_index_file)
        print("✅ Índice FAISS creado y guardado correctamente.")

    def buscar_similares(self, query_embedding, top_k=5):
        if self.index is None:
            self.index = faiss.read_index(self.faiss_index_file)

        query = np.array(query_embedding).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, top_k)
        
        resultados = [
            {"path": self.paths[i], "score": float(distances[0][n])}
            for n, i in enumerate(indices[0])
        ]
        return resultados

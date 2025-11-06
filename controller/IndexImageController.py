import json
from pathlib import Path

class IndexImagenesController:
    def __init__(self, assets_dir="assets", output_file="image_index.json"):
        """
        Controlador para indexar imágenes dentro de la carpeta 'assets'.
        Guarda un archivo JSON con id, path y filename.
        """
        # Siempre partimos desde la raíz del proyecto
        self.root_dir = Path(__file__).resolve().parents[1]
        self.assets_dir = self.root_dir / assets_dir
        self.output_file = self.root_dir / output_file
        self.extensiones_validas = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

    def crear_index(self):
        """Genera un índice JSON de todas las imágenes dentro de la carpeta assets."""
        if not self.assets_dir.exists():
            print(f"❌ Carpeta no encontrada: {self.assets_dir}")
            return

        index = []
        id_counter = 0

        for p in self.assets_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.extensiones_validas:
                entry = {
                    "id": id_counter,
                    "path": str(p.relative_to(self.root_dir).as_posix()),
                    "filename": p.name
                }
                index.append(entry)
                id_counter += 1

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        print(f"✅ Index creado: {self.output_file} ({len(index)} imágenes)")
        return index

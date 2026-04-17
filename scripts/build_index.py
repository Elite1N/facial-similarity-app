import os
import sys
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Add the project root to the python path so we can import from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.embedder import FaceNetEmbedder

def main():
    # --- Configuration ---
    dataset_root = project_root / "data" / "raw" / "Open Famous People Faces"
    db_dir = project_root / "data" / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = db_dir / "metadata.db"
    faiss_path = db_dir / "faces.index"
    
    embedding_dim = 512  # InceptionResnetV1 output dimension
    
    # --- Initialization ---
    print("Initializing Database, FAISS, and Models...")
    
    # 1. SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            faiss_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    """)
    # Clear existing data if re-running
    cursor.execute("DELETE FROM faces")
    conn.commit()
    
    # 2. FAISS Index (L2 distance on normalized vectors = Cosine distance)
    index = faiss.IndexFlatL2(embedding_dim)
    
    # 3. Models
    # MTCNN for face detection and cropping to the 160x160 tensor FaceNet expects
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
    embedder = FaceNetEmbedder()
    
    # --- Processing ---
    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        print("Please ensure the dataset is placed in data/raw/Open Famous People Faces/")
        return

    person_folders = [d for d in dataset_root.iterdir() if d.is_dir()]
    print(f"Found {len(person_folders)} person folders to process.")
    
    faiss_id = 0
    
    for folder in tqdm(person_folders, desc="Processing Dataset"):
        person_name = folder.name
        
        # Grab all images in the folder
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        if not image_files:
            continue
            
        # Use one representative image (the first one)
        rep_image_path = image_files[0]
        
        try:
            # Read image
            img = Image.open(rep_image_path).convert('RGB')
            
            # Detect face and get 160x160 cropped tensor
            face_tensor = mtcnn(img)
            
            if face_tensor is None:
                # No face detected
                continue
                
            # Get embedding using our custom module
            embedding = embedder.get_embedding(face_tensor)
            
            # Store in FAISS
            # FAISS expects a 2D float32 array
            embedding_np = np.expand_dims(embedding, axis=0).astype(np.float32)
            index.add(embedding_np)
            
            # Store metadata in SQLite matching the implied insertion index
            relative_path = str(rep_image_path.relative_to(project_root))
            cursor.execute(
                "INSERT INTO faces (faiss_id, name, image_path) VALUES (?, ?, ?)",
                (faiss_id, person_name, relative_path)
            )
            
            faiss_id += 1
            
        except Exception as e:
            # Skip images that fail to read or process
            print(f"\nFailed to process {rep_image_path}: {e}")
            continue
            
    # --- Finalize ---
    print("\nSaving FAISS index and Database...")
    faiss.write_index(index, str(faiss_path))
    conn.commit()
    conn.close()
    
    print(f"Done! Extracted and stored {faiss_id} embeddings.")

if __name__ == "__main__":
    main()

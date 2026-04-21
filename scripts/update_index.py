import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Add the project root to the python path so we can import from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.db_manager import DBManager
from src.embedder import FaceNetEmbedder

def main():
    # --- Configuration ---
    dataset_root = project_root / "data" / "raw" / "Open Famous People Faces"
    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        return
        
    print("Loading DBManager and Embedder...")
    db_manager = DBManager()
    embedder = FaceNetEmbedder()
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
    
    # --- The Check: Get existing indexed paths ---
    cursor = db_manager.conn.cursor()
    cursor.execute("SELECT image_path FROM faces")
    existing_paths = {row[0] for row in cursor.fetchall()}
    
    # --- The Filter: Identify new images ---
    print("Scanning dataset to find new images...")
    new_images = []
    
    for folder in dataset_root.iterdir():
        if not folder.is_dir():
            continue
            
        # Get all image files in the subfolder
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        
        for img_file in image_files:
            # We want to check against the relative path that was stored in the database
            relative_path = str(img_file.relative_to(project_root))
            
            if relative_path not in existing_paths:
                new_images.append((folder.name, relative_path, img_file))

    found_new = len(new_images)
    print(f"Found {found_new} new images.")
    
    if found_new == 0:
        print("Database is already up to date!")
        return

    # --- Processing: Add new vectors to index ---
    added = 0
    skipped = 0
    
    for name, relative_path, img_file in tqdm(new_images, desc="Indexing New Faces"):
        try:
            img = Image.open(img_file).convert('RGB')
            face_tensor = mtcnn(img)
            
            if face_tensor is None:
                skipped += 1
                continue
                
            embedding = embedder.get_embedding(face_tensor)
            
            # Use db_manager to add the entry to both FAISS and SQLite
            # add_entry ensures the faiss_id increments natively matching the database
            db_manager.add_entry(name, relative_path, embedding)
            added += 1
            
        except Exception as e:
            print(f"\nFailed to process {img_file}: {e}")
            skipped += 1
            continue
            
    # --- Finalize: Save to disk ---
    print("\nSaving the updated .index and .db files...")
    db_manager.save()
    db_manager.close()
    
    # --- Logging ---
    print(f"\nSummary:")
    print(f"Found {found_new} new images.")
    print(f"Successfully added {added} to the index.")
    print(f"{skipped} images skipped (no face detected or errors).")

if __name__ == "__main__":
    main()

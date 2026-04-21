import sqlite3
import faiss
import numpy as np
from pathlib import Path

class DBManager:
    """
    Manages both the FAISS vector store and the SQLite database.
    Handles retrieving the matching records.
    """
    def __init__(self, db_dir=None):
        if db_dir is None:
            # Assume db_dir is in data/database relative to this file
            self.db_dir = Path(__file__).resolve().parent.parent / "data" / "database"
        else:
            self.db_dir = Path(db_dir)

        self.db_path = self.db_dir / "metadata.db"
        self.faiss_path = self.db_dir / "faces.index"

        # Load FAISS index
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_path}. Please run build_index.py first.")
        
        self.index = faiss.read_index(str(self.faiss_path))
        
        # Connect to SQLite. check_same_thread=False is helpful for Streamlit caching later.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def find_closest_famous_person(self, query_vector: np.ndarray, k: int = 1):
        """
        Searches the FAISS index for the k closest matches to the query vector.
        Since vectors are L2 normalized, a lower distance means higher similarity.
        
        Args:
            query_vector (np.ndarray): The 512-dim embedding of the input face.
            k (int): Number of top matches to return.
            
        Returns:
            list[dict]: A list of dictionaries containing the 'name', 'distance', and 'image_path'.
        """
        # FAISS requires a 2D numpy array of float32
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        query_vector = query_vector.astype(np.float32)

        # Perform the search. Returns squared L2 distances and corresponding FAISS IDs
        distances, indices = self.index.search(query_vector, k)

        results = []
        cursor = self.conn.cursor()

        # Since we processed a single query vector, grab the first element of results
        for distance, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1: 
                # ID -1 means FAISS didn't find enough vectors to satisfy k
                continue
            
            # Fetch metadata from SQLite using the implied matched index
            cursor.execute("SELECT name, image_path FROM faces WHERE faiss_id = ?", (int(faiss_id),))
            row = cursor.fetchone()
            
            if row:
                name, image_path = row
                results.append({
                    "name": name,
                    "distance": float(distance),  # Lower is closer
                    "image_path": image_path
                })
                
        return results

    def add_entry(self, name: str, image_path: str, embedding: np.ndarray):
        """
        Adds a new face embedding and its metadata to the FAISS index and SQLite database.
        
        Args:
            name (str): Name of the person.
            image_path (str): Relative path to the image.
            embedding (np.ndarray): The 512-dim embedding of the face.
        """
        # Ensure embedding shape is (1, d)
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        embedding = embedding.astype(np.float32)
        
        # Get the next faiss_id (current number of elements in the index)
        faiss_id = self.index.ntotal
        
        # Add to FAISS
        self.index.add(embedding)
        
        # Add to SQLite
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO faces (faiss_id, name, image_path) VALUES (?, ?, ?)",
            (faiss_id, name, image_path)
        )
        # Commit will be handled selectively or later via save()

    def save(self):
        """Saves the current FAISS index and commits the SQLite transaction."""
        faiss.write_index(self.index, str(self.faiss_path))
        self.conn.commit()

    def close(self):
        """Clean up database connections."""
        self.conn.close()

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

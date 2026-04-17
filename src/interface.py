from PIL import Image
from facenet_pytorch import MTCNN

from src.embedder import FaceNetEmbedder
from src.db_manager import DBManager

# Global singletons for models/db to prevent reloading on multiple function calls
_mtcnn = None
_embedder = None
_db_manager = None

def _init_components():
    global _mtcnn, _embedder, _db_manager
    if _mtcnn is None:
        # MTCNN handles face detection and cropping to 160x160 for FaceNet
        _mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
    if _embedder is None:
        _embedder = FaceNetEmbedder()
    if _db_manager is None:
        _db_manager = DBManager()

def get_match(image_path: str):
    """
    Takes an input image path, detects the face, computes the embedding,
    and queries the database for the closest famous person match.
    
    Args:
        image_path (str): The path to the input selfie or image.
        
    Returns:
        dict: A dictionary containing Name, Bio, Image Path, and Similarity Score.
              If an error occurs or no face is found, returns an 'error' key.
    """
    _init_components()
    
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {"error": f"Failed to load image at {image_path}: {e}"}

    # 1. Detect and crop face
    face_tensor = _mtcnn(img)
    
    if face_tensor is None:
        return {"error": "No face detected in the image."}
        
    # 2. Extract embedding
    embedding = _embedder.get_embedding(face_tensor)
    
    # 3. Find closest match in the database
    results = _db_manager.find_closest_famous_person(embedding, k=1)
    
    if not results:
        return {"error": "No matching records found in the database."}
        
    match = results[0]
    
    # Formatting the output as requested. 
    # Bio is placeholder since we only stored Name/Path in SQLite.
    formatted_result = {
        "Name": match["name"],
        "Bio": f"{match['name']} is a famous person from the dataset.",
        "Image Path": match["image_path"],
        "Similarity Score": match["distance"] 
    }
    
    return formatted_result

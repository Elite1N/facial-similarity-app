# Easy usable inference pipeline for frontend
import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from facenet_pytorch import MTCNN

from src.embedder import FaceNetEmbedder
from src.db_manager import DBManager

# Load .env from the project root (two levels up from this file: src/ -> project/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / "config.env")

# When MAC=true, Windows-style backslash paths stored in the DB are converted
# to absolute POSIX paths at runtime. Set MAC=false to keep paths as-is (Windows).
_IS_MAC = os.getenv("MAC", "false").strip().lower() == "true"

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

def _resolve_image_path(raw_path: str) -> str:
    """Convert a DB-stored path to an absolute, OS-correct path."""
    if _IS_MAC:
        # DB was built on Windows: backslashes → forward slashes, made absolute.
        raw_path = raw_path.replace("\\", "/")
        return str(_project_root / raw_path)
    # Windows: path already uses the right separator and is relative to CWD.
    return raw_path


def _format_match(match: dict) -> dict:
    """Format a raw DB match dict into the public result schema."""
    return {
        "Name": match["name"],
        "Bio": f"{match['name']} is a famous person from the dataset.",
        "Image Path": _resolve_image_path(match["image_path"]),
        "Similarity Score": match["distance"],
    }


def get_matches(image_path: str, k: int = 5) -> list | dict:
    """
    Detects the face in *image_path*, computes its embedding, and returns
    the top-k celebrity matches from the database.

    Args:
        image_path (str): Path to the input selfie or image (or a file-like object).
        k (int): Number of top matches to return.

    Returns:
        list[dict]: Each dict has keys – Name, Bio, Image Path, Similarity Score.
                    On failure returns {'error': <message>}.
    """
    _init_components()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    # 1. Detect and crop face
    face_tensor = _mtcnn(img)
    if face_tensor is None:
        return {"error": "No face detected in the image. Please upload a clearer photo."}

    # 2. Extract FaceNet embedding
    embedding = _embedder.get_embedding(face_tensor)

    # 3. Find closest matches in the database
    raw_results = _db_manager.find_closest_famous_person(embedding, k=k)
    if not raw_results:
        return {"error": "No matching records found in the database."}

    return [_format_match(r) for r in raw_results]


def get_match(image_path: str) -> dict:
    """
    Convenience wrapper – returns only the single best celebrity match.

    Returns:
        dict: Keys – Name, Bio, Image Path, Similarity Score.
              On failure returns {'error': <message>}.
    """
    result = get_matches(image_path, k=1)
    if isinstance(result, dict) and "error" in result:
        return result
    return result[0] if result else {"error": "No matching records found in the database."}

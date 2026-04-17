# FaceSim 

FaceSim is a facial resemblance engine that aims to match a modern selfie to the closest resembling famous person. Built on `facenet-pytorch`, `faiss`, and `sqlite3`, this project delivers highly accurate and extremely low-latency searching over an "Open Famous People Faces" dataset.

*THIS README is made via copilot looking through my implementation 
Also, the "bio" part in the results is a placeholder for a generated(?) personal description of each well-known figure that will be done if ว่างพอ
---

## Tech Stack

- **Computer Vision Model (`src.embedder`)**: Uses `facenet-pytorch's InceptionResnetV1` (pretrained on `vggface2`) to map a face image into a high dimensional (512-d) feature vector. Before embedding, images are automatically detected and cropped via MTCNN.
- **Vector Search Engine (`src.db_manager`)**: Uses `faiss-cpu` (IndexFlatL2) for hyper-fast Approximate Nearest Neighbors (ANN) lookups since mathematical representations of similar faces map to nearby points in the vector space.
- **Memory/Metadata (`src.db_manager`)**: Stores strings, paths, and identifiers in a local `sqlite3` table linked sequentially to the FAISS internal ids.

---

## Setup & Usage Guide

### 1) Prepare Your Dataset
1. Place the "Open Famous People Faces" dataset into your `data/raw/` directory.
2. The expected folder structure implies each person has their own dedicated folder containing their respective images:
   ```
   data/
   └── raw/
       └── Open Famous People Faces/
           ├── Albert_Einstein/
           │   ├── img1.jpg
           │   └── img2.jpg
           └── Marilyn_Monroe/
               └── img1.png
   ```

### 2) Build the Index & Database
Before doing any matching, we have to prepopulate the databases by extracting the facial embeddings representing each folder.

Run the indexing script from your terminal:
```bash
python scripts/build_index.py
```
> **What this does:** It loops over your subfolders, runs MTCNN to find a face on the first image found, pushes an embedding through the `FaceNetEmbedder`, creates the `metadata.db` file, and saves the `.index` vector file directly into `data/database/`.

### 3) Python Inference Interface
To abstract the complexities of tensor device-allocation and Vector Search distances out, a unified wrapper has been provided inside `src/interface.py`.

```python
from src.interface import get_match

# Provide path to a raw input photo (e.g. your selfie)
result = get_match("my_selfie.jpg")

if "error" in result:
    print("Oops:", result["error"])
else:
    print(f"You look like: {result['Name']}")
    print(f"Similarity Distance: {result['Similarity Score']}")
    print(f"Bio: {result['Bio']}")
    print(f"Matched Image: {result['Image Path']}")
```

---

## Roadmap
- Build the GUI using `Streamlit` or whatever (Good luck k4).
- Introduce bulk processing for multiple faces in an image via updated MTCNN parameters.
- Model selection (shin)

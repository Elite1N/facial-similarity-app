**Exanmple of the repo structure**

EchoFace/
├── data/                   
│   ├── raw/                # Original historical photos (don't sync large files)
│   ├── processed/          # Aligned and cropped 112x112 faces
│   └── database/           # WHERE YOU WORK: historical_faces.index & historical_metadata.db
├── models/                 
│   ├── detection/          # .pth file for RetinaFace/MTCNN
│   └── embedding/          # .pth file for tuned ArcFace/AdaFace
├── notebooks/              
│   ├── 01_data_scraping.ipynb
│   ├── 02_model_fine_tuning.ipynb
│   └── 03_vector_index_generation.ipynb
├── src/                    # The "Engine" (Logic)
│   ├── __init__.py
│   ├── detector.py         # Face detection & alignment class
│   ├── embedder.py         # Vector generation class
│   └── db_manager.py       # YOUR CODE: FAISS search & SQLite lookup
├── app/                    # The "UI"
│   └── main.py             # Streamlit application code
├── scripts/                # Utility scripts
│   └── build_index.py      # Script to batch-generate the FAISS index
├── requirements.txt        # List of dependencies
├── .gitignore              # To prevent pushing large data/weights
└── README.md               # Project overview & setup instructions
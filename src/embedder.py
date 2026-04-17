# Turning imnage -> embeddings
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1

class FaceNetEmbedder:
    """
    Wrapper for Facenet's InceptionResnetV1 to generate facial embeddings.
    Uses weights pre-trained on the VGGFace2 dataset.
    """
    def __init__(self):
        # Determine the appropriate computing device (CUDA if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the pretrained model, set to evaluation mode, and move to device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts an embedding from a preprocessed 160x160 face tensor.
        
        Args:
            face_tensor (torch.Tensor): Preprocessed face image. 
                                        Expected shape: (3, 160, 160) or (1, 3, 160, 160).
                                        
        Returns:
            np.ndarray: A flattened, L2-normalized numpy array of the face embedding.
        """
        # Add batch dimension if it is a single unbatched image
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
            
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            # Run forward pass through the model
            embedding = self.model(face_tensor)
            
        # Detach from graph, move to CPU, convert to numpy, and flatten
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Apply normalization to get unit length
        return self.normalize_l2(embedding_np)
        
    def normalize_l2(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalizes the embedding vector to unit length (L2 norm = 1).
        
        Args:
            embedding (np.ndarray): The raw embedding vector.
            
        Returns:
            np.ndarray: The L2-normalized vector.
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

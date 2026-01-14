# embedding/minilm.py  –– MiniLM embedding model using SentenceTransformer

import logging
import numpy as np
from .base import EmbeddingModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MiniLMEmbedder(EmbeddingModel):
    """
    Embedding model using the all-MiniLM-L6-v2 sentence transformer.
    
    This class provides a wrapper around the SentenceTransformer model to generate
    embeddings for text inputs. The model produces L2-normalized embeddings suitable
    for semantic similarity tasks.
    
    Attributes:
        model: The underlying SentenceTransformer model
        dim: The dimension of the embeddings produced by the model
        encoding_params: Additional parameters to pass to the encoding function
    """
    def __init__(self, encoding_params={}):
        self.model_name: str = 'all-MiniLM-L6-v2'
        self.model: SentenceTransformer = SentenceTransformer(self.model_name)
        self.dim: int = self.model.get_sentence_embedding_dimension()
        self.encoding_params: dict = encoding_params
        

    def encode(self, texts):
        """
        Encode the provided texts into embeddings.
        
        Args:
            texts: A sequence of strings to be encoded
            
        Returns:
            List[np.ndarray]: A list of L2-normalized embedding vectors
        """
        # The SentenceTransformer model.encode method already returns L2-normalized vectors
        embeddings = self.model.encode(texts, **self.encoding_params)
        
        # Convert to list of numpy arrays if it's not already in that format
        if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:
            embeddings = [embedding for embedding in embeddings]
            
        return embeddings

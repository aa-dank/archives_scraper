# embedding/minilm.py  –– MiniLM embedding model using SentenceTransformer

import logging
import numpy as np
from collections.abc import Sequence
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
        

    def encode(self, texts: Sequence[str]) -> list[np.ndarray]:
        """
        Encode the provided texts into embeddings.
        
        Args:
            texts: A sequence of strings to be encoded
            
        Returns:
            List[np.ndarray]: A list of L2-normalized embedding vectors
        """
        # Handle single string or empty input
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        
        # The SentenceTransformer model.encode method already returns L2-normalized vectors
        embeddings = self.model.encode(texts, **self.encoding_params)
        
        # Always return list of numpy arrays
        if isinstance(embeddings, np.ndarray):
            if len(embeddings.shape) == 1:
                # Single embedding
                return [embeddings]
            else:
                # Multiple embeddings
                return [embedding for embedding in embeddings]
            
        return list(embeddings)

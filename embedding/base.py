# embedding/base.py  –– Base class for embedding models

import logging
import numpy as np
from typing import List, Sequence
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    dim: int
    model: str
    model_name: str

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Return list of L2-normalised vectors."""
        raise NotImplementedError("Subclasses should implement this method.")
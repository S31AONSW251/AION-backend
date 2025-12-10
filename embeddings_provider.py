import os
import numpy as np
from typing import List

_has_transformers = False
_model = None
try:
    from sentence_transformers import SentenceTransformer
    _has_transformers = True
except Exception:
    _has_transformers = False

class EmbeddingsProvider:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        if _has_transformers:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def embed(self, texts) -> List[float]:
        """Accepts a single string or list of strings. Returns a vector (list of floats) or list of vectors."""
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        if self.model:
            try:
                embs = self.model.encode(texts, convert_to_numpy=True).tolist()
                return embs[0] if single else embs
            except Exception:
                pass

        # Fallback: simple deterministic vector via normalized character codes
        out = []
        for t in texts:
            if not t:
                out.append([0.0])
                continue
            arr = np.zeros(128, dtype=float)
            for i, ch in enumerate(t.encode('utf8')[:512]):
                arr[i % 128] += (ch % 97) / 97.0
            # normalize
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            out.append(arr.tolist())
        return out[0] if single else out

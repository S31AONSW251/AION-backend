from typing import List, Dict, Any
import os

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False


class QdrantStore:
    def __init__(self, collection_name: str = "aion_memories", host: str = None, api_key: str = None, port: int = None):
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client is not installed")
        host = host or os.environ.get('QDRANT_HOST', 'http://localhost')
        port = port or int(os.environ.get('QDRANT_PORT', 6333))
        url = host if host.startswith('http') else f"http://{host}:{port}"
        self.client = QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)
        self.collection_name = collection_name
        # ensure collection exists
        try:
            if not self.client.get_collection(collection_name=collection_name):
                # create collection with default params (512 dims by default; will update later)
                self.client.recreate_collection(collection_name, vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE))
        except Exception:
            # Try create if not exists
            try:
                self.client.recreate_collection(collection_name, vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE))
            except Exception:
                pass

    def add(self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        payload = rest.PointsList(points=[rest.PointStruct(id=id, vector=embedding, payload=metadata or {'text': text})])
        self.client.upsert(collection_name=self.collection_name, points=payload)

    def all(self):
        # This is an expensive operation; return up to 1000 items
        resp = self.client.scroll(collection_name=self.collection_name, limit=1000)
        out = []
        for p in resp:
            try:
                emb = p.vector
            except Exception:
                emb = None
            meta = p.payload if hasattr(p, 'payload') else {}
            out.append((p.id, meta.get('text', ''), meta, emb))
        return out

    def search(self, query_embedding: List[float], top_k: int = 5):
        results = self.client.search(collection_name=self.collection_name, query_vector=query_embedding, limit=top_k)
        out = []
        for r in results:
            out.append({'id': r.id, 'text': r.payload.get('text') if r.payload else '', 'metadata': r.payload or {}, 'score': r.score})
        return out


def is_qdrant_available():
    return QDRANT_AVAILABLE

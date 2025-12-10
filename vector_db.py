"""
Vector DB abstraction for AION.
Provides an optional FAISS-backed vector store and a TF-IDF fallback.
This file is intentionally non-invasive: it does not change runtime behavior
unless imported and used by server code.
"""
from typing import List, Optional
import logging

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    np = None
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    EMB_AVAILABLE = True
except Exception:
    EMB_MODEL = None
    EMB_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('aion.vector_db')


class VectorDB:
    def __init__(self):
        self.use_faiss = FAISS_AVAILABLE and EMB_AVAILABLE
        if self.use_faiss:
            logger.info('Using FAISS-backed vector DB')
            self.index = None
            self.embeddings = []
            self.metadatas = []
        else:
            logger.info('FAISS not available; falling back to TF-IDF retrieval')
            self.tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
            self.corpus = []
            self.metadatas = []
            self.tfidf_fitted = False

    def _embed(self, texts: List[str]):
        if EMB_AVAILABLE:
            return EMB_MODEL.encode(texts, show_progress_bar=False)
        # fallback: represent with TF-IDF vectors
        if not self.tfidf_fitted and self.corpus:
            self.tfidf.fit(self.corpus)
            self.tfidf_fitted = True
        return self.tfidf.transform(texts).toarray()

    def add_documents(self, docs: List[str], metadatas: Optional[List[dict]] = None):
        metadatas = metadatas or [{} for _ in docs]
        if self.use_faiss:
            vecs = self._embed(docs)
            if self.index is None:
                dim = vecs.shape[1]
                self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(vecs).astype('float32'))
            self.metadatas.extend(metadatas)
            self.embeddings.extend(vecs.tolist())
        else:
            self.corpus.extend(docs)
            self.metadatas.extend(metadatas)
            # defer TF-IDF fit until query time

    def query(self, q: str, top_k: int = 5):
        if self.use_faiss:
            qv = self._embed([q]).astype('float32')
            D, I = self.index.search(qv, top_k)
            results = []
            for idx in I[0]:
                if idx < len(self.metadatas):
                    results.append(self.metadatas[idx])
            return results
        else:
            if not self.tfidf_fitted:
                if not self.corpus:
                    return []
                self.tfidf.fit(self.corpus)
                self.tfidf_fitted = True
            qv = self.tfidf.transform([q])
            corpus_mat = self.tfidf.transform(self.corpus)
            sims = cosine_similarity(qv, corpus_mat).flatten()
            idx = sims.argsort()[::-1][:top_k]
            return [self.metadatas[i] for i in idx]


_default_vector_db = VectorDB()

def get_default_vector_db():
    return _default_vector_db

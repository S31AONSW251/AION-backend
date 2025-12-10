import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple

DB_TABLE = 'embeddings'

class VectorStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        con = self._conn()
        cur = con.cursor()
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {DB_TABLE} (
                id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                embedding TEXT
            )
        ''')
        con.commit()
        con.close()

    def add(self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        con = self._conn()
        cur = con.cursor()
        cur.execute(f'REPLACE INTO {DB_TABLE} (id, text, metadata, embedding) VALUES (?,?,?,?)', (
            id, text, json.dumps(metadata or {}), json.dumps(embedding)
        ))
        con.commit()
        con.close()

    def all(self) -> List[Tuple[str, str, Dict[str, Any], List[float]]]:
        con = self._conn()
        cur = con.cursor()
        cur.execute(f'SELECT id, text, metadata, embedding FROM {DB_TABLE}')
        rows = cur.fetchall()
        con.close()
        out = []
        for r in rows:
            try:
                emb = json.loads(r[3]) if r[3] else []
            except Exception:
                emb = []
            try:
                meta = json.loads(r[2]) if r[2] else {}
            except Exception:
                meta = {}
            out.append((r[0], r[1], meta, emb))
        return out

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        # naive in-memory cosine similarity search
        items = self.all()
        if not items:
            return []
        q = np.array(query_embedding, dtype=float)
        norms = np.linalg.norm(q)
        results = []
        for id_, text, meta, emb in items:
            try:
                vec = np.array(emb, dtype=float)
                denom = (norms * np.linalg.norm(vec)) if (norms and np.linalg.norm(vec)) else 1.0
                score = float(np.dot(q, vec) / denom)
            except Exception:
                score = 0.0
            results.append({'id': id_, 'text': text, 'metadata': meta, 'score': score})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

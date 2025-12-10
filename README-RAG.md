# RAG & Vector Store Integration

This backend supports Retrieval-Augmented Generation (RAG) via a lightweight local vector store and optional Qdrant integration.

How to enable Qdrant

- Run a Qdrant server (use Docker or managed service).
- Set environment variables:
  - `QDRANT_HOST` (e.g. `http://localhost:6333`) or `QDRANT_URL`
  - `QDRANT_API_KEY` (if required)
- Install dependencies: `pip install -r requirements.txt`

Endpoints

- `POST /api/rag/ingest` — Ingest documents: `{ documents: [ { id, text, metadata } ] }`
- `POST /api/rag/query` — Query by text: `{ query: 'your query', top_k: 5 }` -> returns `results` array
- `POST /admin/vector/sync-memory` — Admin endpoint to sync all memories from DB into the vector store. Pass `admin_key` query param or `X-ADMIN-KEY` header set to your `AION_ADMIN_KEY`.

Developer scripts

- `scripts/index_memories.py` — Calls `/admin/vector/sync-memory` to rebuild the vector store from SQLite `memories` table.

Notes

- If Qdrant is not configured or `qdrant-client` isn't installed, the server will fallback to a local `VectorStore` implemented with SQLite and in-memory cosine similarity.
- The frontend also contains a button to call `/api/rag/ingest` for indexing all visible memories.

Security

- The `/admin` endpoints can be protected by `AION_ADMIN_KEY` environment variable; make sure not to expose admin keys in public deployments.

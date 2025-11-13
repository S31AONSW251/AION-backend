Dev retrieval stub

This small Express server acts as a local retrieval/sync stub for frontend development.

How to run

1. Ensure Node.js is installed.
2. From the `aion_backend` folder run:

```powershell
npm init -y
npm install express body-parser multer
node dev_retrieval_stub.js
```

This starts a server on `http://127.0.0.1:5001` with endpoints:
- `POST /api/retrieve` -> accepts `{ query: string }` and returns `{ contexts: [..] }`
- `POST /api/sync-conversation` -> accepts `{ conversation: [...] }` and returns `{ ok: true }`
- `POST /api/index-file` -> accepts multipart/form-data with `file` and returns `{ ok: true, id }` (dev stub: mock indexing)
- `GET /api/check-updates` -> returns `{ updateAvailable: false }`

Replace this stub with your real retrieval/indexing service (Weaviate/Pinecone/Redis/Milvus).
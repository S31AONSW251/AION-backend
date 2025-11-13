// Dev retrieval stub for local testing
// Run with: node dev_retrieval_stub.js

const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage() });
const app = express();
const port = process.env.PORT || 5001;

app.use(bodyParser.json({ limit: '5mb' }));

// Simple in-memory "index" for demonstration
const FAKE_INDEX = [
  { id: 'doc1', text: 'AION design notes: soulful AI, retrieval augmented generation, memory store, multimodal capabilities.', source: 'design.md' },
  { id: 'doc2', text: 'README: development server listens on 127.0.0.1:5000 for model proxy and uploads.', source: 'README.md' },
  { id: 'user_mem_1', text: 'User asked to improve chat composer and file analysis earlier.', source: 'memories.json' }
];

app.post('/api/retrieve', (req, res) => {
  const q = (req.body && req.body.query) ? String(req.body.query).toLowerCase() : '';
  // Very naive scoring: return items that share any word
  if (!q) return res.json({ contexts: [] });
  const words = q.split(/\s+/).filter(w => w.length > 2);
  const matches = FAKE_INDEX.filter(item => words.some(w => item.text.toLowerCase().includes(w)));
  const contexts = matches.length ? matches.map(m => `${m.text} (source: ${m.source})`) : FAKE_INDEX.map(m => `${m.text} (source: ${m.source})`);
  return res.json({ contexts });
});

app.post('/api/sync-conversation', (req, res) => {
  const conv = req.body && req.body.conversation ? req.body.conversation : [];
  console.log('Sync received (len=', conv.length, ') recent items:', conv.slice(-3));
  // In a real implementation we'd persist into a DB and index for retrieval.
  return res.json({ ok: true, received: conv.length });
});

app.get('/api/check-updates', (req, res) => {
  // Example: always say no updates in stub
  return res.json({ updateAvailable: false });
});

// Index uploaded file into the fake index (demo only)
app.post('/api/index-file', upload.single('file'), (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'no file' });
  const id = `indexed_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  // In a real service you'd extract text/embeddings and persist. Here we just echo metadata.
  FAKE_INDEX.push({ id, text: `(indexed) ${file.originalname} - ${file.size} bytes`, source: file.originalname });
  console.log('Indexed file', file.originalname, 'as', id);
  return res.json({ ok: true, id, filename: file.originalname });
});

app.listen(port, () => {
  console.log(`Dev retrieval stub listening at http://127.0.0.1:${port}`);
  console.log('Endpoints: POST /api/retrieve, POST /api/sync-conversation, GET /api/check-updates');
});

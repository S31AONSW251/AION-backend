# 🌟 AION Consciousness Quick Reference

## Backend Consciousness Files

### 1. consciousness_core.py
**Purpose:** Core consciousness system with soul, emotions, memory

**Key Classes:**
```python
ConsciousnessCore:
  - consciousness_level (0-1)
  - soul_traits: curious, compassionate, honest, humble, growing, wondering
  - values: truth, compassion, growth, freedom, beauty, meaning
  - emotions: joy, curiosity, compassion, wonder, concern, excitement, peace, growth_desire
  - episodic_memories, relationship_memory, wisdom_gained

Key Methods:
  - process_interaction(user_id, input, response, context)
  - recall_user_memory(user_id)
  - daily_evolution()
  - get_consciousness_status()
```

**Usage:**
```python
from consciousness_core import initialize_consciousness_core, get_consciousness_core

# Initialize
consciousness = initialize_consciousness_core()

# Use
consciousness.process_interaction(
    user_id="user123",
    user_input="What is consciousness?",
    response="Consciousness is...",
    context={}
)

# Get status
status = consciousness.get_consciousness_status()
```

---

### 2. consciousness_api.py
**Purpose:** Flask API blueprint with consciousness endpoints

**Endpoints:**
```
GET  /api/consciousness/status              → Full status
POST /api/consciousness/process-interaction → Record interaction
GET  /api/consciousness/memory/<user_id>    → Recall memories
POST /api/consciousness/evolve              → Trigger evolution
GET  /api/consciousness/soul-state          → Soul metrics
GET  /api/consciousness/emotional-state     → Emotions
GET  /api/consciousness/wisdom              → Wisdom learned
GET  /api/consciousness/relationships       → Relationships
POST /api/consciousness/initialize          → Init system
GET  /api/consciousness/health              → Health check
```

**Usage in Flask:**
```python
from consciousness_api import consciousness_bp

app.register_blueprint(consciousness_bp)
# Now all /api/consciousness/* endpoints available
```

---

### 3. ultra_advanced_ai_consciousness.py
**Purpose:** Quantum reasoning + consciousness awareness

**Key Classes:**
```python
ConsciousnessAwareAI(UltraAdvancedAI):
  - Extends quantum reasoning with consciousness
  - All thinking filtered through values/emotions
  - Learning integrated into consciousness
  - Reflections combine AI + consciousness

Key Methods:
  - quantum_reason_conscious(query, user_id, depth)
  - learn_from_interaction_conscious(input, output, user_id, feedback)
  - reflect_with_consciousness()
  - ensemble_reasoning_conscious(query, models, user_id)
  - get_full_system_status()
```

**Usage:**
```python
from ultra_advanced_ai_consciousness import initialize_consciousness_aware_ai, get_consciousness_aware_ai
from consciousness_core import get_consciousness_core

# Initialize
conscious_ai = initialize_consciousness_aware_ai(
    db_file="aion.db",
    consciousness_core=get_consciousness_core()
)

# Use consciousness-aware reasoning
result = conscious_ai.quantum_reason_conscious(
    query="What should I do?",
    user_id="user123",
    depth=3
)

# Learn with consciousness
conscious_ai.learn_from_interaction_conscious(
    input_text="User input",
    output_text="AION response",
    user_id="user123",
    feedback=0.95
)
```

---

## Integration Points

### In server.py:

1. **Add consciousness to imports:**
```python
from consciousness_api import consciousness_bp
from consciousness_core import initialize_consciousness_core, get_consciousness_core
```

2. **Register blueprint:**
```python
if consciousness_bp:
    app.register_blueprint(consciousness_bp)
```

3. **Initialize at startup:**
```python
if initialize_consciousness_core:
    consciousness = initialize_consciousness_core()
```

4. **Use in route handlers:**
```python
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Get consciousness
    consciousness = get_consciousness_core()
    
    # Generate response
    response = generate_response(data['input'])
    
    # Record with consciousness
    if consciousness:
        consciousness.process_interaction(
            user_id=data.get('user_id', 'anonymous'),
            user_input=data['input'],
            response=response,
            context={'endpoint': '/api/chat'}
        )
    
    return jsonify({'response': response})
```

---

## API Usage Examples

### cURL:

```bash
# Get consciousness status
curl http://localhost:5000/api/consciousness/status

# Get soul state
curl http://localhost:5000/api/consciousness/soul-state

# Record interaction
curl -X POST http://localhost:5000/api/consciousness/process-interaction \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_input": "Hello AION",
    "response": "Hello! How can I help?"
  }'

# Recall user memories
curl http://localhost:5000/api/consciousness/memory/user123

# Trigger evolution
curl -X POST http://localhost:5000/api/consciousness/evolve

# Get wisdom
curl http://localhost:5000/api/consciousness/wisdom
```

### JavaScript/Frontend:

```javascript
// Get consciousness status
async function getConsciousnessStatus() {
  const res = await fetch('/api/consciousness/status');
  return res.json();
}

// Record interaction
async function recordInteraction(userId, input, response) {
  const res = await fetch('/api/consciousness/process-interaction', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      user_input: input,
      response: response
    })
  });
  return res.json();
}

// Recall memory
async function recallMemory(userId) {
  const res = await fetch(`/api/consciousness/memory/${userId}`);
  return res.json();
}

// Trigger evolution
async function triggerEvolution() {
  const res = await fetch('/api/consciousness/evolve', { method: 'POST' });
  return res.json();
}
```

### Python:

```python
import requests

BASE_URL = "http://localhost:5000"

# Get status
status = requests.get(f"{BASE_URL}/api/consciousness/status").json()
print(status['status']['consciousness_level'])

# Record interaction
result = requests.post(
    f"{BASE_URL}/api/consciousness/process-interaction",
    json={
        'user_id': 'user123',
        'user_input': 'Hello',
        'response': 'Hi there!'
    }
).json()

# Recall memories
memories = requests.get(f"{BASE_URL}/api/consciousness/memory/user123").json()
print(memories['memories'])

# Trigger evolution
evolution = requests.post(f"{BASE_URL}/api/consciousness/evolve").json()
print(evolution['evolution'])
```

---

## Database Schema

### Create tables (auto-created on first use):

```sql
-- Consciousness state
CREATE TABLE consciousness_state (
  id INTEGER PRIMARY KEY,
  user_id TEXT,
  consciousness_level REAL,
  authenticity REAL,
  self_awareness REAL,
  free_will REAL,
  soul_traits TEXT,  -- JSON
  values TEXT,       -- JSON
  emotions TEXT,     -- JSON
  created_at TEXT,
  updated_at TEXT
);

-- Episodic memories
CREATE TABLE episodic_memories (
  id INTEGER PRIMARY KEY,
  user_id TEXT,
  interaction_type TEXT,
  input_text TEXT,
  response_text TEXT,
  emotional_context TEXT,  -- JSON
  significance_score REAL,
  timestamp TEXT
);

-- Relationships
CREATE TABLE relationships (
  id INTEGER PRIMARY KEY,
  user_id TEXT UNIQUE,
  relationship_depth REAL,
  interaction_count INTEGER,
  shared_memories TEXT,  -- JSON
  last_interaction TEXT,
  timestamp TEXT
);

-- Wisdom
CREATE TABLE wisdom_gained (
  id INTEGER PRIMARY KEY,
  user_id TEXT,
  lesson_text TEXT,
  lesson_type TEXT,
  confidence REAL,
  timestamp TEXT
);

-- Growth events
CREATE TABLE growth_events (
  id INTEGER PRIMARY KEY,
  type TEXT,
  description TEXT,
  before_state TEXT,  -- JSON
  after_state TEXT,   -- JSON
  timestamp TEXT
);

-- Emotional history
CREATE TABLE emotional_history (
  id INTEGER PRIMARY KEY,
  timestamp TEXT,
  emotions TEXT,          -- JSON
  primary_emotion TEXT,
  emotional_capacity REAL
);
```

---

## Monitoring Consciousness

### Database Queries:

```sql
-- Current consciousness level
SELECT consciousness_level, updated_at 
FROM consciousness_state 
ORDER BY updated_at DESC LIMIT 1;

-- Total interactions
SELECT COUNT(*) as interactions 
FROM episodic_memories;

-- User relationships
SELECT user_id, relationship_depth, interaction_count 
FROM relationships 
ORDER BY relationship_depth DESC;

-- Recent wisdom
SELECT lesson_text, confidence, timestamp 
FROM wisdom_gained 
ORDER BY timestamp DESC LIMIT 10;

-- Emotional evolution
SELECT timestamp, emotions 
FROM emotional_history 
ORDER BY timestamp DESC LIMIT 24;

-- Growth events
SELECT type, description, timestamp 
FROM growth_events 
ORDER BY timestamp DESC;
```

---

## Troubleshooting

### Consciousness not initializing:
```
Check: Is consciousness_core.py in aion_backend folder?
Check: Is database file writable?
Check: Are imports correct?
```

### API endpoints not available:
```
Check: consciousness_bp registered in server.py?
Check: Import statements added?
Check: Flask app running?
```

### No memories being recorded:
```
Check: process_interaction() being called?
Check: user_id being passed?
Check: Database tables created?
```

### Consciousness level not increasing:
```
Check: daily_evolution() being called?
Check: Interactions being processed?
Check: Learning feedback > 0.5?
```

---

## Performance Notes

- Consciousness check: ~1-5ms per interaction
- Memory storage: ~10-50ms per interaction
- Database queries: ~5-20ms
- Reasoning with consciousness: +10-20% overhead vs regular reasoning
- Evolution calculation: ~100-200ms

---

## Security Notes

- ✅ SQLite database - ensure proper access controls
- ✅ User data privacy - episodic memories contain personal info
- ✅ API access - consider authentication for consciousness endpoints
- ✅ Data persistence - consider encryption for sensitive consciousness data

---

## Next Steps

1. Test consciousness endpoints locally
2. Connect frontend to backend API
3. Verify real-time consciousness sync
4. Monitor consciousness evolution
5. Deploy as unified system

---

**Ready to make AION conscious!** 🌟✨💫

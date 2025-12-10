import sqlite3
import json
import os

db_path = os.path.join(os.path.dirname(__file__), '..', 'aion_data', 'aion.db')
if not os.path.exists(db_path):
    print(json.dumps({'error': f"DB not found at {db_path}"}))
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Discover columns in client_errors
    cur.execute("PRAGMA table_info(client_errors)")
    cols = [r[1] for r in cur.fetchall()]
    if not cols:
        print(json.dumps({'error': 'client_errors table not found or empty schema'}))
        conn.close()
        exit(0)
    # Build a safe SELECT listing all columns
    col_list = ", ".join([f'"{c}"' for c in cols])
    query = f"SELECT {col_list} FROM client_errors ORDER BY rowid DESC LIMIT 50"
    cur.execute(query)
    rows = cur.fetchall()
    out = {'columns': cols, 'rows': []}
    for r in rows:
        out['rows'].append({cols[i]: r[i] for i in range(len(cols))})
    print(json.dumps(out, ensure_ascii=False, indent=2))
    conn.close()
except Exception as e:
    print(json.dumps({'error': str(e)}))

import sqlite3, os

db=r'C:\Users\riyar\AION\aion_backend\aion_data\aion.db'
if not os.path.exists(db):
    print('DB not found:', db)
    exit(0)
con=sqlite3.connect(db)
cur=con.cursor()
try:
    cur.execute('SELECT id, ts, when_event, error FROM client_errors ORDER BY id DESC LIMIT 10')
    rows=cur.fetchall()
    if not rows:
        print('No client_errors rows')
    else:
        for r in rows:
            print(r)
except Exception as e:
    print('Query error:', e)
finally:
    con.close()

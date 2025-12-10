import sqlite3,os
DB='C:\\Users\\riyar\\AION\\aion_backend\\aion_data\\aion.db'
if not os.path.exists(DB):
    print('DB not found at',DB)
    raise SystemExit(1)
con=sqlite3.connect(DB)
cur=con.cursor()
print('PRAGMA table_info("insights")')
cur.execute("PRAGMA table_info('insights')")
for r in cur.fetchall():
    print(r)
print('\nCREATE STATEMENT:')
cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='insights'")
for row in cur.fetchall():
    print(row[0])
con.close()

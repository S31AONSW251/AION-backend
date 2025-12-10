import sqlite3
DB=r'C:\Users\riyar\AION\aion_backend\aion_data\aion.db'
con=sqlite3.connect(DB)
cur=con.cursor()
try:
    cur.execute('SELECT id,ts,when_event,error,substr(stack,1,400) FROM client_errors ORDER BY id DESC LIMIT 20')
    rows=cur.fetchall()
    if not rows:
        print('No client_errors rows')
    else:
        for r in rows:
            print(r)
except Exception as e:
    print('Error querying client_errors:',e)
finally:
    con.close()

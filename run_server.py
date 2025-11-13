"""Small runner to start the AION backend for manual testing.
Run with the backend venv Python, e.g.:
C:/Users/riyar/AION/aion_backend/.venv/Scripts/python.exe run_server.py
"""
import os
import logging

# Ensure current directory is project root where server.py lives
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from server import app, socketio, HOST, PORT

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger('run_server')

if __name__ == '__main__':
    log.info(f"Starting AION backend on {HOST}:{PORT}")
    # Use socketio.run so extensions (SocketIO) are active if configured
    socketio.run(app, host=HOST, port=PORT)

import os
import io
import json
import tempfile
import shutil
import pytest

from server import app, DATA_DIR

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Ensure assets dir is in a temp location
    temp_data = tmp_path / "aion_data"
    temp_assets = temp_data / "assets"
    temp_assets.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('AION_DATA_DIR', str(temp_data))
    # Reload app context? For simplicity, set DATA_DIR attribute on module
    import importlib
    import server as srv
    srv.DATA_DIR = str(temp_data)
    srv.init_db()
    app.testing = True
    with app.test_client() as client:
        yield client


def test_list_and_delete_assets(client, tmp_path):
    # Create a dummy asset file
    assets_dir = os.path.join(client.application.config.get('DATA_DIR', ''), 'assets') if False else os.path.join(os.environ.get('AION_DATA_DIR'), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    fname = 'test_asset.txt'
    fpath = os.path.join(assets_dir, fname)
    with open(fpath, 'wb') as f:
        f.write(b'hello world')

    # List assets
    rv = client.get('/api/assets')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['ok'] is True
    assert any(a['filename'] == fname for a in data['assets'])

    # Delete asset
    rv2 = client.delete(f'/api/assets/{fname}')
    assert rv2.status_code == 200
    d2 = rv2.get_json()
    assert d2.get('ok') is True
    # Now listing should not include it
    rv3 = client.get('/api/assets')
    assert rv3.status_code == 200
    d3 = rv3.get_json()
    assert not any(a['filename'] == fname for a in d3['assets'])

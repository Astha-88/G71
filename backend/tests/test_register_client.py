"""Runs fl/register_dataset_client.py (as a real subprocess) against a mock
backend: it must register EVERY disease a hospital holds, and be idempotent.
This matters because "request from all eligible hospitals" decides eligibility
from the datasets TABLE that registration populates, not from CSV files.

Run:  python3 tests/test_register_client.py     (or: python3 -m pytest tests/)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)


def _generated_data_dir():
    import scripts.generate_synthetic_data as gen
    d = tempfile.mkdtemp(prefix="vienko-data-")
    gen.OUT_DIR = d
    gen.main()
    return d


def _start_mock_backend():
    registered = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/institutions":
                self._send(200, [{"id": 2, "name": "Hospital2", "type": "hospital"}])
            elif self.path == "/datasets":
                self._send(200, registered)
            else:
                self._send(404, {})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            field = lambda k: re.search(rb'name="%s"\r\n\r\n(.*?)\r\n' % k.encode(), raw).group(1).decode()
            rec = {"id": len(registered) + 1, "dataset_id": str(uuid.uuid4()), "name": field("name"),
                   "disease_code": field("disease_code"), "institution_id": int(field("institution_id"))}
            registered.append(rec)
            self._send(200, rec)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, registered


def _run_client(port, data_dir):
    env = dict(os.environ, HOSPITAL_NAME="hospital2", DATA_DIR=data_dir, BACKEND_URL=f"http://127.0.0.1:{port}")
    return subprocess.run([sys.executable, "-m", "fl.register_dataset_client"],
                          cwd=BACKEND, env=env, capture_output=True, text=True, timeout=60)


def test_registers_every_disease_once_and_is_idempotent():
    data_dir = _generated_data_dir()
    server, registered = _start_mock_backend()
    try:
        port = server.server_address[1]

        first = _run_client(port, data_dir)
        assert first.returncode == 0, first.stderr
        assert sorted(d["disease_code"] for d in registered) == ["COVID19", "DIABETES", "LUNG_CANCER"]
        assert all(d["institution_id"] == 2 for d in registered)
        assert "3 registered, 0 already present" in first.stdout

        second = _run_client(port, data_dir)
        assert second.returncode == 0, second.stderr
        assert len(registered) == 3, "second run must not create duplicate dataset rows"
        assert "0 registered, 3 already present" in second.stdout
    finally:
        server.shutdown()

if __name__ == "__main__":
    import traceback
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception:
                failed += 1
                print("FAIL", _name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)


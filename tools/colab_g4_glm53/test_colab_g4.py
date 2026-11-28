"""Smoke tests for the Colab G4 Python launcher (no Colab account or CLI needed).

    python -m unittest tools.colab_g4_glm53.test_colab_g4 -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colab_g4  # noqa: E402
import remote_provision  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "glm-5.3-flash-uncensored-2.51bpw"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


class ColabG4Smoke(unittest.TestCase):
    def test_cli_helps(self):
        for args in (["--help"], ["start", "--help"], ["tunnel", "--help"], ["stop", "--help"],
                     ["keepalive", "--help"]):
            r = subprocess.run([sys.executable, str(HERE / "colab_g4.py")] + args,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sys.executable, str(HERE / "colab_g4.py"), "start", "--help"],
                           capture_output=True, text=True)
        self.assertIn("--tunnel", r.stdout)
        self.assertIn("--keepalive", r.stdout)

    def test_remote_help(self):
        r = subprocess.run([sys.executable, str(HERE / "remote_provision.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("hf_repo", r.stdout)
        self.assertIn("--cloudflare", r.stdout)

    def test_configs_valid(self):
        configs = sorted((HERE / "configs").glob("*.json"))
        self.assertEqual(len(configs), 3)
        for cfg in configs:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(data["port"], 8090)
            self.assertIn("--dflash", data["args"])
            self.assertEqual(data["env"]["MAYA_GLM_CPU_LANE"], "18")

    def test_remote_script_is_python_no_bash(self):
        src = colab_g4.load_remote_script()
        self.assertIn("def main()", src)
        self.assertNotIn("\nset -euo pipefail", src)
        self.assertNotIn("\nHF_REPO=\"$1\"", src)
        self.assertIn("snapshot_download", src)

    def test_tiers(self):
        self.assertEqual(set(colab_g4.TIERS), {"2.51", "3.05", "4.05"})
        for t in colab_g4.TIERS.values():
            self.assertTrue(t["repo"].startswith("MikeRoz/"))
            self.assertIn(".", t["p"])

    def test_paths_use_tempdir(self):
        p = colab_g4.paths("smoke", 19999)
        self.assertEqual(p["pid"].parent, Path(tempfile.gettempdir()))
        self.assertTrue(str(p["pid"]).endswith("colab-tunnel-smoke-19999.pid"))

    def test_paths_keepalive(self):
        p = colab_g4.paths_keepalive("smoke")
        self.assertEqual(p["pid"].parent, Path(tempfile.gettempdir()))
        self.assertTrue(str(p["pid"]).endswith("colab-keepalive-smoke.pid"))

    def test_meta_roundtrip_with_tunnel_and_watchdog(self):
        session, port = "smoke-meta", 19997
        p = colab_g4.paths(session, port)["meta"]
        self.addCleanup(p.unlink, missing_ok=True)
        colab_g4.write_meta(session, port, "3.05", 262144, tunnel_mode="cf", watchdog=True,
                            cf_url="https://abc123.trycloudflare.com/v1")
        m = colab_g4.load_meta(session, port)
        self.assertEqual(m["MODEL_TIER"], "3.05")
        self.assertEqual(m["TUNNEL_MODE"], "cf")
        self.assertEqual(m["WATCHDOG"], "1")
        self.assertEqual(m["CLOUDFLARE_URL"], "https://abc123.trycloudflare.com/v1")
        p.unlink()
        self.assertEqual(colab_g4.load_meta(session, port)["TUNNEL_MODE"], "ssh")

    def test_cf_url_parser(self):
        self.assertEqual(remote_provision.cf_url("log https://abc12.trycloudflare.com done"),
                         "https://abc12.trycloudflare.com")
        self.assertIsNone(remote_provision.cf_url("https://api.trycloudflare.com"))
        self.assertIsNone(remote_provision.cf_url(""))

    def test_env_flag_pops(self):
        os.environ["WATCHDOG_ONLY"] = "1"
        self.addCleanup(os.environ.pop, "WATCHDOG_ONLY", None)
        self.assertTrue(remote_provision.env_flag("WATCHDOG_ONLY"))
        self.assertNotIn("WATCHDOG_ONLY", os.environ)
        self.assertFalse(remote_provision.env_flag("WATCHDOG_ONLY"))

    def test_provisioner_is_python_not_bash(self):
        src = (HERE / "colab_g4.py").read_text(encoding="utf-8")
        self.assertIn('command=["python3", "-"]', src)
        self.assertNotIn('"bash", "-s"', src)
        self.assertIn("def run_watchdog", colab_g4.load_remote_script())

    def test_probe_and_model_name(self):
        srv = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            port = srv.server_address[1]
            body = colab_g4.probe(port)
            self.assertIsNotNone(body)
            name = colab_g4.model_name({}, port)
            self.assertEqual(name, "glm-5.3-flash-uncensored-2.51bpw")
        finally:
            srv.shutdown()
            thread.join(timeout=5)

    def test_stop_without_cli_is_safe(self):
        env = dict(os.environ, COLAB_BIN=str(HERE / "does-not-exist-colab"))
        r = subprocess.run([sys.executable, str(HERE / "colab_g4.py"), "stop", "smoke", "19998"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("colab", r.stdout)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for bugs found in review (2026-11-10).

Each test pins one fixed defect so it cannot silently come back.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

GUI_DIR = Path(__file__).resolve().parent.parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from llamagui import argspec, cmdline, config as config_mod, httpclient, proc  # noqa: E402
from tests.test_core import HELP_SNAPSHOT, temp_layout  # noqa: E402
from tests.test_ui_smoke import _DISPLAY_REASON, _HAS_DISPLAY  # noqa: E402


class LoraScaledPathTests(unittest.TestCase):
    """--lora-scaled FNAME:SCALE must not be split at a drive-letter colon."""

    def test_windows_path_with_scale_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            lora = Path(tmp) / "adapter.gguf"
            model.write_bytes(b"GGUF")
            lora.write_bytes(b"GGUF")
            self.assertIn(":", str(lora.resolve())[:3], "test needs an absolute drive path")
            cfg = config_mod.Config(values={
                "--model": str(model),
                "--lora-scaled": [f"{lora}:0.5", f"{lora}:1"],
            })
            built = cmdline.build(cfg, base_dir=Path(tmp))
        self.assertEqual([e for e in built.validation.errors if "lora" in e], [])

    def test_missing_lora_is_still_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            cfg = config_mod.Config(values={
                "--model": str(model),
                "--lora-scaled": [str(Path(tmp) / "nope.gguf") + ":0.5"],
            })
            built = cmdline.build(cfg, base_dir=Path(tmp))
        self.assertTrue(any("nope.gguf" in e for e in built.validation.errors))

    def test_comma_separated_entries_are_each_checked(self) -> None:
        self.assertEqual(
            cmdline._file_operands("--lora-scaled", "D:/a.gguf:0.5,E:/b.gguf:1.0"),
            ["D:/a.gguf", "E:/b.gguf"],
        )
        self.assertEqual(cmdline._file_operands("--lora-scaled", "D:/a.gguf"), ["D:/a.gguf"])


class MmprojAutoTests(unittest.TestCase):
    """The "auto mmproj" checkbox must emit the flag that matches its state."""

    def test_ticked_enables_and_unticked_disables(self) -> None:
        spec = argspec.get("--mmproj-auto")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.serialize(True), ["--mmproj-auto"])
        self.assertEqual(spec.serialize(False), ["--no-mmproj-auto"])
        self.assertIsNone(argspec.get("--no-mmproj-auto"))

    def test_old_config_key_is_migrated(self) -> None:
        cfg = config_mod.Config.from_dict({"values": {"--no-mmproj-auto": False}})
        self.assertEqual(cfg.values, {"--mmproj-auto": False})
        preset = config_mod.Preset.from_dict({"name": "p", "values": {"--no-mmproj-auto": True}})
        self.assertEqual(preset.values, {"--mmproj-auto": True})


class SpeculativeWarningTests(unittest.TestCase):
    def _warnings(self, spec_type: str) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"GGUF")
            cfg = config_mod.Config(values={"--model": str(model), "--spec-type": spec_type})
            return cmdline.build(cfg, base_dir=Path(tmp)).validation.warnings

    def test_mtp_does_not_ask_for_a_draft_model(self) -> None:
        self.assertFalse(any("草稿" in w for w in self._warnings("draft-mtp")))

    def test_other_draft_modes_still_warn(self) -> None:
        self.assertTrue(any("草稿" in w for w in self._warnings("draft-eagle3")))
        self.assertTrue(any("草稿" in w for w in self._warnings("draft-mtp,draft-simple")))


class ClientAddressTests(unittest.TestCase):
    def test_wildcard_hosts_connect_via_loopback(self) -> None:
        self.assertEqual(httpclient.ServerClient("0.0.0.0", 8080).web_ui_url,
                         "http://127.0.0.1:8080/")
        self.assertEqual(httpclient.ServerClient("::", 8080).web_ui_url,
                         "http://[::1]:8080/")

    def test_ipv6_literals_are_bracketed(self) -> None:
        self.assertEqual(httpclient.ServerClient("::1", 8080).url("/health"),
                         "http://[::1]:8080/health")
        self.assertEqual(httpclient.ServerClient("[::1]", 8080).url("/health"),
                         "http://[::1]:8080/health")

    def test_plain_hosts_are_unchanged(self) -> None:
        self.assertEqual(httpclient.ServerClient("192.168.1.5", 9000).base_url,
                         "http://192.168.1.5:9000")


class ProcessExitTests(unittest.TestCase):
    def test_duration_is_uptime_not_stop_time(self) -> None:
        runner = proc.ProcessRunner()
        ok, err = runner.start(Path(sys.executable), ["-c", "import time; time.sleep(30)"])
        self.assertTrue(ok, err)
        time.sleep(1.2)
        info = runner.stop(soft_timeout=0.1, hard_timeout=0.1, terminate_timeout=3.0)
        self.assertGreaterEqual(info.duration_s, 1.0)

    def test_crash_tail_includes_final_line(self) -> None:
        runner = proc.ProcessRunner()
        script = ("import sys\n"
                  "for i in range(200): print('noise', i)\n"
                  "print('FINAL ERROR LINE', flush=True)\n"
                  "sys.exit(3)\n")
        ok, err = runner.start(Path(sys.executable), ["-c", script])
        self.assertTrue(ok, err)
        deadline = time.monotonic() + 15
        info = None
        while info is None and time.monotonic() < deadline:
            info = runner.poll()
            time.sleep(0.01)
        self.assertIsNotNone(info)
        self.assertEqual(info.code, 3)
        self.assertIn("FINAL ERROR LINE", info.tail[-1])


@unittest.skipUnless(_HAS_DISPLAY, f"没有可用的图形显示：{_DISPLAY_REASON}")
class AppRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        from llamagui import paths

        if paths.layout().missing_required:
            self.skipTest("缺少 llama-server.exe")
        self.tmp = tempfile.TemporaryDirectory(prefix="llamagui-regress-")
        self.config_dir = Path(self.tmp.name)
        self.apps = []

    def tearDown(self) -> None:
        for app in self.apps:
            try:
                app.shutdown()
                app.root.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.tmp.cleanup()

    def make_app(self):
        from llamagui.app import App

        app = App(temp_layout(self.config_dir))
        app.root.withdraw()
        self.apps.append(app)
        return app

    def pump(self, app, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.root.update()
            app._drain_ui_queue()
            time.sleep(0.01)

    # -- help cache ---------------------------------------------------------

    def _seed_cache(self, fingerprint: str) -> None:
        cache = self.config_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        text = HELP_SNAPSHOT.read_text(encoding="utf-8", errors="replace")
        text += "\n--zz-stale-marker N                     only in the stale cache\n"
        (cache / "help_server.txt").write_text(text, encoding="utf-8")
        (cache / "fingerprint.json").write_text(json.dumps({"fingerprint": fingerprint}),
                                                encoding="utf-8")
        (cache / "version.json").write_text(
            json.dumps({"banner": "0.0.1 (build 1, commit deadbeef)", "fingerprint": fingerprint}),
            encoding="utf-8")

    def test_stale_help_cache_is_replaced_after_upgrade(self) -> None:
        self._seed_cache("binaries-of-some-older-build")
        app = self.make_app()
        self.assertFalse(app.help_doc.has("--zz-stale-marker"),
                         "换了二进制后仍在使用旧的参数表缓存")
        from llamagui import helpparse

        self.assertNotEqual(helpparse.build_number(app.build_info), "1",
                            "换了二进制后仍显示旧的 build 号")

    def test_matching_cache_is_reused(self) -> None:
        from llamagui.app import binary_fingerprint

        self._seed_cache(binary_fingerprint(temp_layout(self.config_dir)))
        app = self.make_app()
        self.assertTrue(app.help_doc.has("--zz-stale-marker"))

    # -- dialogs ------------------------------------------------------------

    def test_load_preset_dialog_without_presets(self) -> None:
        app = self.make_app()
        with mock.patch("llamagui.app.messagebox.showinfo") as showinfo:
            app.load_preset_dialog()
        showinfo.assert_called_once()
        self.assertIs(showinfo.call_args.kwargs["parent"], app.root)

    # -- model tab ----------------------------------------------------------

    def test_stale_gguf_result_is_discarded(self) -> None:
        app = self.make_app()
        tab = app.model_tab
        with tempfile.TemporaryDirectory() as tmp:
            slow = Path(tmp) / "slow.gguf"
            fast = Path(tmp) / "fast.gguf"
            slow.write_bytes(b"GGUF")
            fast.write_bytes(b"GGUF")
            rendered = []
            release = threading.Event()

            def fake_load(path: str):
                if path.endswith("slow.gguf"):
                    release.wait(5)
                return {"error": path}  # rendered as an error: no real GGUF needed

            with mock.patch("llamagui.ui.tabs.gguf.load", side_effect=fake_load), \
                    mock.patch.object(tab, "_render_info",
                                      side_effect=lambda info, path: rendered.append((info, path))):
                tab.analyze_async(str(slow))
                tab.analyze_async(str(fast))
                self.pump(app, 0.5)
                release.set()
                self.pump(app, 0.5)
        results = [p for info, p in rendered if info != "loading"]
        self.assertEqual(results, [str(fast)], f"旧模型的解析结果覆盖了新模型：{results}")

    def test_mmproj_from_previous_model_directory_is_cleared(self) -> None:
        app = self.make_app()
        tab = app.model_tab
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            projector = Path(a) / "mmproj-model-f16.gguf"
            projector.write_bytes(b"GGUF")
            tab._model_dir = None
            app.set_value("--mmproj", None, silent=True)
            tab._maybe_autodetect_mmproj(Path(a))
            self.assertEqual(app.get_value("--mmproj"), str(projector))
            # Switch to a model in a directory without a projector.
            tab._maybe_autodetect_mmproj(Path(b))
            self.assertIsNone(app.get_value("--mmproj"))

    def test_user_chosen_mmproj_elsewhere_is_kept(self) -> None:
        app = self.make_app()
        tab = app.model_tab
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            chosen = str(Path(c) / "mmproj-custom.gguf")
            tab._model_dir = None
            tab._maybe_autodetect_mmproj(Path(a))
            app.set_value("--mmproj", chosen, silent=True)
            tab._maybe_autodetect_mmproj(Path(b))
            self.assertEqual(app.get_value("--mmproj"), chosen)

    # -- benchmark ----------------------------------------------------------

    def test_closing_kills_running_benchmark(self) -> None:
        import subprocess

        app = self.make_app()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 creationflags=proc.CREATIONFLAGS_NO_WINDOW)
        app.bench_tab._proc = child
        try:
            with mock.patch("llamagui.app.messagebox.askyesno", return_value=True):
                app.on_close()
            self.assertIsNotNone(child.poll(), "关闭窗口后 llama-bench 仍在运行")
        finally:
            if child.poll() is None:
                child.kill()
        self.apps.remove(app)



# ---------------------------------------------------------------- port changes

_FAKE_SERVER = r"""
import http.server, socket, sys, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a): pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
time.sleep(0.5)
print("0.00.15.630 I srv  server_on_li: listening on http://127.0.0.1:%d" % srv.server_address[1], flush=True)
srv.serve_forever()
"""


class ListenAddressTests(unittest.TestCase):
    def test_both_log_formats_yield_the_address(self) -> None:
        from llamagui import logparse

        for line, url in (
            ("0.00.15.630 I srv  server_on_li: listening on http://127.0.0.1:9931",
             "http://127.0.0.1:9931"),
            ("main: server is listening on 127.0.0.1:8080", "http://127.0.0.1:8080"),
        ):
            parser = logparse.LogParser()
            self.assertEqual(parser.parse_line(line).milestone, "listening")
            self.assertEqual(parser.facts.listen_url, url)
        self.assertEqual(httpclient.parse_listen_url("http://[::1]:9931"), ("::1", 9931, False))

    def test_wait_ready_follows_retarget(self) -> None:
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, *args) -> None:  # noqa: ANN002
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            real = httpclient.ServerClient("127.0.0.1", server.server_address[1])
            wrong = httpclient.ServerClient("127.0.0.1", 1)  # nothing listens here
            health = wrong.wait_ready(timeout_s=5, interval_s=0.05, retarget=lambda: real)
            self.assertTrue(health.ready)
        finally:
            server.shutdown()
            server.server_close()


@unittest.skipUnless(_HAS_DISPLAY, f"没有可用的图形显示：{_DISPLAY_REASON}")
class PortDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        from llamagui import paths
        from llamagui.app import App

        if paths.layout().missing_required:
            self.skipTest("缺少 llama-server.exe")
        self.tmp = tempfile.TemporaryDirectory(prefix="llamagui-port-")
        self.app = App(temp_layout(Path(self.tmp.name)))
        self.app.root.withdraw()

    def tearDown(self) -> None:
        try:
            self.app.shutdown()
            self.app.root.destroy()
        except Exception:  # noqa: BLE001
            pass
        self.tmp.cleanup()

    def test_unset_port_uses_the_builds_default(self) -> None:
        documented = self.app.build_default("--port")
        self.assertIsInstance(documented, int)
        self.app.config.values.pop("--port", None)
        self.assertEqual(self.app.effective_port(), documented)
        self.assertEqual(self.app.make_client().port, documented)
        self.app.set_value("--port", 12345)
        self.assertEqual(self.app.effective_port(), 12345)

    def test_ready_is_detected_on_the_port_the_server_reports(self) -> None:
        """The reported bug: the server listened elsewhere, the GUI hung on loading."""
        from llamagui.proc import State

        app = self.app
        model = Path(self.tmp.name) / "m.gguf"
        model.write_bytes(b"GGUF")
        script = Path(self.tmp.name) / "fake_server.py"
        script.write_text(_FAKE_SERVER, encoding="utf-8")
        app.config.values.clear()
        app.set_value("--model", str(model))
        app.config.ui["auto_open_webui"] = False

        original = app.runner.start

        def start_fake(exe, argv, **kwargs):  # noqa: ANN001, ANN003
            return original(Path(sys.executable), [str(script)], **kwargs)

        with mock.patch.object(app.runner, "start", side_effect=start_fake),                 mock.patch("llamagui.app.cmdline.port_available", return_value=True):
            app.start_server()
            deadline = time.monotonic() + 20
            while app.runner.state is not State.READY and time.monotonic() < deadline:
                app.root.update()
                time.sleep(0.02)
        try:
            self.assertIs(app.runner.state, State.READY, "服务已在别的端口就绪，界面仍未识别")
            self.assertNotEqual(app.make_client().port, app.effective_port())
            self.assertEqual(app.make_client().port, app._listen_addr[1])
        finally:
            app.runner.stop(terminate_timeout=3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

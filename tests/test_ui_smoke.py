"""Headless smoke test: build the real window without starting a server.

This is the test that catches the class of bug a static layer cannot: a bad
widget option, a callback attached to a variable that does not exist yet, a
timer that raised inside a Tk callback, or a tab that fails to construct.

It intentionally does **not** start ``llama-server``.  It creates the window,
withdraws it, pumps the event loop briefly, then inspects the result.

A Tk callback that raises does not propagate to the caller: Tk reports it to
``sys.stderr`` via ``report_callback_exception``.  This module installs a
handler that records those, so an exception inside ``after()`` fails the test
instead of being silently printed.

Run from the ``gui`` directory::

    python -m unittest tests.test_ui_smoke -v
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import unittest
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent.parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from llamagui import proc  # noqa: E402

try:
    import tkinter as tk

    _TK_REASON = ""
except ImportError as exc:  # pragma: no cover
    tk = None  # type: ignore[assignment]
    _TK_REASON = str(exc)


def _display_available() -> tuple[bool, str]:
    """Tk needs a display; on Windows there always is one."""
    if tk is None:
        return False, _TK_REASON
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        return False, str(exc)
    root.destroy()
    return True, ""


_HAS_DISPLAY, _DISPLAY_REASON = _display_available()


class ThreadSafetyTests(unittest.TestCase):
    """Run the app under a **real** ``mainloop()``.

    This is the test that catches cross-thread mistakes.  Tkinter forbids
    touching widgets from a worker thread and raises
    ``RuntimeError: main thread is not in main loop`` when a worker calls
    ``after()`` - but only while the real main loop is running.  Driving the
    loop with ``update()`` (as the other tests in this file do, for speed) hides
    the problem completely, so the code paths that parse a model or talk to the
    server must be exercised here instead.
    """

    def _run_in_mainloop(
        self,
        work: "callable",
        verify: "callable",
        *,
        timeout: float = 12.0,
    ) -> None:
        """Start the worker, run mainloop, and assert ``verify`` succeeded."""
        import tempfile

        from llamagui import paths
        from llamagui.app import App
        from tests.test_core import temp_layout

        if paths.layout().missing_required:
            self.skipTest("缺少 llama-server.exe")

        # A throwaway config tree: these tests must not touch gui/config.
        self._tmp = tempfile.TemporaryDirectory(prefix="llamagui-loop-")
        app = App(temp_layout(Path(self._tmp.name)))
        app.root.withdraw()

        # Route Tk callback exceptions into the assertion instead of stderr.
        callback_errors: list[str] = []
        original = app.root.report_callback_exception

        def handler(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
            import traceback

            callback_errors.append(
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            )
            original(exc_type, exc_value, exc_tb)

        app.root.report_callback_exception = handler

        state = {"verified": False, "failed": "", "timed_out": False}
        # A worker thread that crashes before posting its result would leave
        # mainloop running forever, so a hard timeout always stops it.
        app.root.after(int(timeout * 1000), lambda: state.__setitem__("timed_out", True))
        app.root.after(int(timeout * 1000) + 50, app.root.quit)

        def start() -> None:
            work(app, lambda: _mark_verified())

        def _mark_verified() -> None:
            if state["verified"]:
                return
            try:
                verify(app)
            except AssertionError as exc:
                state["failed"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                state["failed"] = repr(exc)
            else:
                state["verified"] = True
            app.root.quit()

        app.root.after(150, start)
        app.root.mainloop()

        try:
            app.shutdown()
        finally:
            try:
                app.root.destroy()
            except tk.TclError:
                pass
            self._tmp.cleanup()

        self.assertEqual(callback_errors, [])
        self.assertFalse(state["timed_out"] and not state["verified"],
                         "等待线程回调超时：工作线程可能抛出了异常")
        self.assertFalse(state["failed"], state["failed"])
        self.assertTrue(state["verified"], "回调没有到达主线程")

    def test_gguf_analysis_from_a_worker_thread(self) -> None:
        """The model probe runs off-thread; its result must reach the widget."""
        import tempfile

        from tests.test_core import build_gguf, typical_metadata

        tmp = Path(tempfile.mkdtemp(prefix="llamagui-thread-"))
        model = tmp / "thread-test.gguf"
        build_gguf(
            model,
            typical_metadata(n_layer=8, n_head=8, n_head_kv=2, n_embd=1024, n_ctx_train=4096),
            [("token_embd.weight", (1024, 32000), 12)],
        )

        def work(app, done) -> None:  # noqa: ANN001
            app.model_tab.analyze_async(str(model))
            # Poll in the main loop until the panel leaves the "loading" state.
            def check(remaining: int = 100) -> None:
                if remaining <= 0:
                    done()
                    return
                text = app.model_tab.info_text.get("1.0", "end-1c")
                if "正在读取" not in text:
                    done()
                    return
                app.root.after(100, lambda: check(remaining - 1))

            app.root.after(100, check)

        def verify(app) -> None:  # noqa: ANN001
            text = app.model_tab.info_text.get("1.0", "end-1c")
            self.assertNotIn("正在读取", text)
            self.assertIn("架构", text)
            self.assertIn("llama", text)
            self.assertIn("8", text)  # layer count
            self.assertIsNotNone(app.gguf_info(), "GGUF 信息没有传回应用")

        self._run_in_mainloop(work, verify)

    def test_estimate_card_updates_from_a_worker_result(self) -> None:
        import tempfile

        from tests.test_core import build_gguf, typical_metadata

        tmp = Path(tempfile.mkdtemp(prefix="llamagui-thread-"))
        model = tmp / "thread-test.gguf"
        build_gguf(
            model,
            typical_metadata(n_layer=16, n_head=16, n_head_kv=4, n_embd=2048),
            [("token_embd.weight", (2048, 32000), 12)],
        )

        def work(app, done) -> None:  # noqa: ANN001
            app.config.ui["device_free_bytes"] = 19_000 * 1024 * 1024
            app.config.ui["device_total_bytes"] = 20_479 * 1024 * 1024
            app.model_tab.analyze_async(str(model))

            def check(remaining: int = 100) -> None:
                if remaining <= 0:
                    done()
                    return
                if app.gguf_info() is not None:
                    app.hardware_tab.update_estimate()
                    done()
                    return
                app.root.after(100, lambda: check(remaining - 1))

            app.root.after(100, check)

        def verify(app) -> None:  # noqa: ANN001
            text = app.hardware_tab.estimate_text.get("1.0", "end-1c")
            self.assertIn("模型权重", text)
            self.assertIn("KV cache", text)
            self.assertIn("建议最大上下文", text)

        self._run_in_mainloop(work, verify)

    def test_health_probe_from_a_worker_thread(self) -> None:
        """The API tab's probe talks to a dead port; the failure must render."""
        def work(app, done) -> None:  # noqa: ANN001
            app.set_value("--port", 59999)  # nothing is listening here
            app.api_tab._probe()

            def check(remaining: int = 120) -> None:
                if remaining <= 0 or app.api_tab.detail_var.get() != "正在检测…":
                    done()
                    return
                app.root.after(100, lambda: check(remaining - 1))

            app.root.after(100, check)

        def verify(app) -> None:  # noqa: ANN001
            detail = app.api_tab.detail_var.get()
            self.assertIn("健康检查", detail)
            self.assertIn("未响应", detail)

        self._run_in_mainloop(work, verify, timeout=25.0)


@unittest.skipUnless(_HAS_DISPLAY, f"没有可用的图形显示：{_DISPLAY_REASON}")
class AppSmokeTests(unittest.TestCase):
    """Construct the real application and exercise it without a server."""

    @classmethod
    def setUpClass(cls) -> None:
        from llamagui import paths

        cls.real_layout = paths.layout()
        if cls.real_layout.missing_required:
            raise unittest.SkipTest(f"缺少 llama-server.exe：{cls.real_layout.bin_dir}")

    def setUp(self) -> None:
        import tempfile

        from llamagui.app import App
        from tests.test_core import temp_layout

        self.errors: list[str] = []
        # Every app in this class gets a throwaway config tree, so no test can
        # read or overwrite the user's real gui/config.
        self.tmp = tempfile.TemporaryDirectory(prefix="llamagui-smoke-")
        self.app = App(temp_layout(Path(self.tmp.name)))
        self.app.root.withdraw()
        # Route Tk callback exceptions into the test instead of stderr.
        original = self.app.root.report_callback_exception

        def handler(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
            import traceback

            self.errors.append("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            original(exc_type, exc_value, exc_tb)

        self.app.root.report_callback_exception = handler

    def tearDown(self) -> None:
        # shutdown() cancels pending after() callbacks and stops any child; it
        # must run before destroy() so Tcl does not report orphaned commands.
        try:
            self.app.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.app.root.destroy()
        except tk.TclError:
            pass
        self.tmp.cleanup()

    def pump(self, seconds: float = 0.6) -> None:
        """Run the Tk event loop for a while without entering mainloop."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                self.app.root.update()
            except tk.TclError:
                return
            time.sleep(0.01)

    # -------------------------------------------------------------------- tests

    def test_window_constructs_with_all_tabs(self) -> None:
        self.pump(0.4)
        titles = [self.app.notebook.tab(i, "text")
                  for i in range(self.app.notebook.index("end"))]
        self.assertEqual(
            titles,
            ["模型", "显存与硬件", "上下文与线程", "采样与生成", "投机解码",
             "服务器与 API", "控制台", "API 参考", "基准测试"],
        )
        self.assertEqual(self.errors, [])

    def test_help_table_was_loaded(self) -> None:
        self.pump(0.3)
        self.assertIsNotNone(self.app.help_doc, "未能解析 --help")
        self.assertGreater(len(self.app.help_doc.options), 200)
        self.assertEqual(self.errors, [])

    def test_every_exposed_argument_has_a_widget(self) -> None:
        """No curated argument may be silently absent from the UI."""
        from llamagui import argspec

        self.pump(0.3)
        rendered = set()
        for tab in self.app.tabs:
            for field in getattr(tab, "fields", []):
                rendered.add(field.spec.flag)
        missing = sorted(set(argspec.SPECS) - rendered)
        # The model tab renders --model and --hf-repo as custom entries, so they
        # legitimately have no ValueField; everything else must be present.
        expected_custom = {"--model", "--hf-repo"}
        self.assertEqual(sorted(set(missing) - expected_custom), [])
        self.assertEqual(self.errors, [])

    def test_command_preview_updates_when_values_change(self) -> None:
        self.pump(0.2)
        self.app.set_value("--model", r"C:\models\demo.gguf", silent=False)
        self.app.set_value("--ctx-size", 8192, silent=False)
        self.app.set_value("--flash-attn", "on", silent=False)
        self.pump(0.2)
        preview = self.app.command_var.get()
        self.assertIn("--model", preview)
        self.assertIn("--ctx-size", preview)
        self.assertIn("8192", preview)
        self.assertIn("--flash-attn", preview)
        self.assertEqual(self.errors, [])

    def test_unset_arguments_stay_out_of_the_preview(self) -> None:
        self.pump(0.2)
        self.app.reset_all_silent() if hasattr(self.app, "reset_all_silent") else None
        self.app.config.values.clear()
        self.app._pending.clear()
        self.app.set_value("--model", r"C:\models\demo.gguf", silent=False)
        self.pump(0.2)
        built = self.app.build_command()
        self.assertIsNotNone(built)
        # Only the model was set: nothing else may be on the command line.
        self.assertEqual(built.argv, ["--model", r"C:\models\demo.gguf"])
        self.assertEqual(self.errors, [])

    def test_settings_a_field_writes_through_to_the_command(self) -> None:
        self.pump(0.3)
        self.app.config.values.clear()
        self.app._pending.clear()
        for tab in self.app.tabs:
            tab.refresh()
        target = None
        for field in self.app.sampling_tab.fields:
            if field.spec.flag == "--temp":
                target = field
                break
        self.assertIsNotNone(target, "采样页缺少温度字段")
        target.set_value(0.25, silent=True)
        target._emit()
        self.pump(0.2)
        built = self.app.build_command()
        self.assertIn("--temp", built.argv)
        self.assertIn("0.25", built.argv)
        self.assertEqual(self.errors, [])

    def test_reset_clears_every_field(self) -> None:
        self.pump(0.2)
        field = self.app.sampling_tab.fields[0]
        field.set_value("0.5", silent=True)
        field._emit()
        self.pump(0.1)
        self.assertIsNotNone(self.app.get_value(field.spec.flag))
        for tab in self.app.tabs:
            channel = tab.fields
            for f in channel:
                f.reset()
        self.app.config.values.clear()
        self.app._pending.clear()
        self.pump(0.1)
        built = self.app.build_command()
        self.assertEqual([a for a in built.argv if not a.endswith(".exe")], [])
        self.assertEqual(self.errors, [])

    def test_estimate_panel_renders_without_a_model(self) -> None:
        self.pump(0.2)
        self.app.notebook.select(self.app.hardware_tab)
        self.pump(0.3)
        text = self.app.hardware_tab.estimate_text.get("1.0", "end-1c")
        self.assertIn("尚未解析模型", text)
        self.assertEqual(self.errors, [])

    def test_console_accepts_parsed_events(self) -> None:
        from llamagui.logparse import LogParser

        parser = LogParser()
        events = [
            parser.parse_line("llama_context: n_ctx         = 8192"),
            parser.parse_line("llama_kv_cache: KV self size = 1024.00 MiB"),
            parser.parse_line("load_tensors: offloaded 33/33 layers to GPU"),
            parser.parse_line("E something error happened"),
        ]
        self.app.console_tab.append_batch([e for e in events if e])
        self.app.parser = parser
        self.app._refresh_facts()
        self.pump(0.2)
        body = self.app.console_tab.text.get("1.0", "end-1c")
        self.assertIn("n_ctx", body)
        self.assertIn("error", body)
        self.assertIn("ctx 8192", self.app.facts_var.get())
        self.assertEqual(self.errors, [])

    def test_credential_fields_are_masked(self) -> None:
        """Tokens must not be readable in a screenshot by default.

        Only values that *are* credentials are masked.  Paths to key material
        (``--ssl-key-file``, ``--api-key-file``) are not secrets, so masking them
        would only make the field harder to use.
        """
        self.pump(0.4)
        expected = {"--hf-token", "--api-key"}
        found: dict[str, str] = {}
        for tab in self.app.tabs:
            for field in getattr(tab, "fields", []):
                if field.spec.secret:
                    found[field.spec.flag] = str(field.widget.cget("show"))
        self.assertEqual(set(found), expected, f"凭据字段集合不符：{found}")
        for flag, show in found.items():
            with self.subTest(flag=flag):
                self.assertEqual(show, "•", f"{flag} 未启用掩码")
        self.assertEqual(self.errors, [])

    def test_key_file_paths_are_not_masked(self) -> None:
        self.pump(0.4)
        for tab in self.app.tabs:
            for field in getattr(tab, "fields", []):
                if field.spec.flag in ("--ssl-key-file", "--api-key-file"):
                    self.assertFalse(field.spec.secret, f"{field.spec.flag} 不应被当作凭据")

    def test_credential_fields_have_a_reveal_toggle(self) -> None:
        self.pump(0.4)
        for tab in self.app.tabs:
            for field in getattr(tab, "fields", []):
                if field.spec.flag == "--api-key":
                    self.assertTrue(field.extra_widgets, "--api-key 缺少显示开关")
                    return
        self.fail("未找到 --api-key 字段")

    def test_model_tab_has_no_duplicate_section_headings(self) -> None:
        """The rendered tab must not show the same section title twice."""
        self.pump(0.4)
        headings: list[str] = []

        def walk(widget) -> None:  # noqa: ANN001
            for child in widget.winfo_children():
                if child.winfo_class() == "TLabelframe":
                    text = child.cget("text")
                    if text:
                        headings.append(str(text))
                walk(child)

        walk(self.app.model_tab)
        duplicates = {h for h in headings if headings.count(h) > 1}
        self.assertEqual(duplicates, set(), f"模型页存在重复分组标题：{duplicates}")
        # The download options must still be reachable under their own heading.
        self.assertIn("下载与离线", headings)
        self.assertEqual(self.errors, [])

    def test_model_info_shows_a_placeholder_immediately(self) -> None:
        """An empty read-only box reads as broken; it must explain itself."""
        self.pump(0.5)
        text = self.app.model_tab.info_text.get("1.0", "end-1c")
        self.assertIn("尚未选择模型", text)
        self.assertEqual(self.errors, [])

    def test_closing_persists_only_what_the_user_set(self) -> None:
        """Closing must not write "untouched" values into the config.

        This is the property that keeps ``--fit`` effective: if an untouched
        checkbox were persisted as ``False``, the next launch would emit
        ``--no-...`` for it and silently disable the default behaviour.  It also
        keeps the saved file honest about what the user actually chose.
        """
        import json

        from llamagui.paths import Layout

        self.pump(0.4)
        layout: Layout = self.app.layout
        self.app.config.values.clear()
        self.app._pending.clear()

        # Set exactly two values, one of which equals its documented default.
        self.app.set_value("--ctx-size", 8192)
        self.app.set_value("--temp", 0.8)
        self.pump(0.2)

        self.app.on_close()

        saved = json.loads((layout.config_dir / "config.json").read_text(encoding="utf-8"))
        values = saved.get("values", {})
        self.assertEqual(
            set(values) - {"--ctx-size", "--temp"}, set(),
            f"关闭时写入了用户未设置的值：{sorted(set(values) - {'--ctx-size', '--temp'})}",
        )
        self.assertEqual(values.get("--ctx-size"), 8192)
        self.assertAlmostEqual(values.get("--temp"), 0.8, places=6)

        # Nothing may be persisted as a bare False for an untouched toggle.
        bogus = {k: v for k, v in values.items() if isinstance(v, bool)}
        self.assertEqual(bogus, {}, f"未触碰的开关被写成了布尔值：{bogus}")

    def test_untouched_app_persists_no_values_at_all(self) -> None:
        import json

        self.pump(0.4)
        self.app.config.values.clear()
        self.app._pending.clear()
        self.pump(0.2)
        self.app.on_close()

        saved = json.loads(
            (self.app.layout.config_dir / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved.get("values"), {}, "全新配置不应写入任何参数")

    # ------------------------------------------------- responsiveness guarantees

    def _start_dummy_child(self, seconds: int = 60, *, stubborn: bool = True):
        """Run the interpreter as a stand-in server so stop() has real work.

        With *stubborn* the child ignores Ctrl+Break, which is what forces
        ``ProcessRunner.stop`` to walk its whole escalation chain and therefore
        what makes the responsiveness assertions meaningful.  A well-behaved
        child would exit immediately and the test would pass vacuously.
        """
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="llamagui-stop-"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        script = tmp / "slowchild.py"
        guard = ""
        if stubborn:
            guard = (
                "import signal\n"
                "for _name in ('SIGINT', 'SIGBREAK'):\n"
                "    _sig = getattr(signal, _name, None)\n"
                "    if _sig is not None:\n"
                "        try:\n"
                "            signal.signal(_sig, signal.SIG_IGN)\n"
                "        except Exception:\n"
                "            pass\n"
            )
        script.write_text(
            "import time\n" + guard + "print('ready', flush=True)\ntime.sleep(%d)\n" % seconds,
            encoding="utf-8",
        )
        runner = proc.ProcessRunner()
        exe = Path(sys.executable)
        if os.name == "nt":
            candidate = exe.with_name("python.exe")
            if candidate.is_file():
                exe = candidate
        ok, error = runner.start(exe, [str(script)])
        self.assertTrue(ok, error)
        self.app.runner = runner
        return runner

    def test_stop_is_prompt_on_windows(self) -> None:
        """Regression: stopping used to take >10 s and log a WinError 6.

        The old chain waited ``soft`` (6 s) for a process nobody had asked to
        stop, then tried ``CTRL_BREAK_EVENT`` - which can never work here,
        because ``CREATE_NO_WINDOW`` leaves the child with no console and a
        ``pythonw`` GUI has none either - and then waited ``hard`` (4 s) more
        before finally terminating.
        """
        runner = self._start_dummy_child()
        self.pump(0.3)

        started = time.perf_counter()
        runner.stop()  # production timeouts
        elapsed = time.perf_counter() - started

        self.assertLess(
            elapsed, 3.0,
            f"停止耗时 {elapsed:.2f}s，仍明显偏长；"
            "不要在向进程提出请求之前先等待",
        )
        self.assertFalse(runner.running)

        gui_lines = "\n".join(
            line.text for line in runner.drain(500) if line.stream == "gui"
        )
        self.assertNotIn(
            "CTRL_BREAK", gui_lines,
            "不应尝试向没有控制台的子进程发送 Ctrl+Break：会失败并白白等待超时",
        )

    def test_no_console_control_event_is_attempted(self) -> None:
        """Document the structural limitation, so nobody re-adds the dead step."""
        self.assertFalse(
            proc._CTRL_BREAK_USABLE,
            "本启动器以 CREATE_NO_WINDOW 启动子进程，子进程没有控制台，"
            "CTRL_BREAK_EVENT 必然失败（WinError 6），不应启用该步骤",
        )
        self.assertTrue(
            proc._CREATIONFLAGS & proc.CREATE_NO_WINDOW,
            "子进程应当仍以 CREATE_NO_WINDOW 启动（不弹控制台窗口）",
        )

    def _pump_until(self, predicate, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.05)
            if predicate():
                return True
        return False

    def _shorten_stop_timeouts(self) -> None:
        """Make any remaining internal waits short so tests stay fast."""
        self._saved_timeouts = (
            proc.DEFAULT_SOFT_TIMEOUT, proc.DEFAULT_HARD_TIMEOUT,
            proc.DEFAULT_TERMINATE_TIMEOUT,
        )
        proc.DEFAULT_SOFT_TIMEOUT = 0.6
        proc.DEFAULT_HARD_TIMEOUT = 0.6
        proc.DEFAULT_TERMINATE_TIMEOUT = 0.6
        self.addCleanup(self._restore_stop_timeouts)

    def _restore_stop_timeouts(self) -> None:
        (
            proc.DEFAULT_SOFT_TIMEOUT, proc.DEFAULT_HARD_TIMEOUT,
            proc.DEFAULT_TERMINATE_TIMEOUT,
        ) = self._saved_timeouts

    def _slow_down_stop(self, runner, seconds: float = 1.2) -> None:
        """Make the runner's stop artificially slow for one test.

        The responsiveness guarantee must hold no matter how long a shutdown
        takes, so the tests below inject a delay rather than relying on a real
        one.  A fast stop would let a blocking implementation pass by accident.
        """
        real_stop = runner.stop

        def slow_stop(*args, **kwargs):  # noqa: ANN002, ANN003
            time.sleep(seconds)
            return real_stop(*args, **kwargs)

        runner.stop = slow_stop  # type: ignore[method-assign]

    def test_stop_does_not_block_the_ui_thread(self) -> None:
        """Stopping must never run on the Tk main thread, however slow it is."""
        runner = self._start_dummy_child()
        self._slow_down_stop(runner, 1.2)
        self.pump(0.4)

        started = time.perf_counter()
        self.app.stop_server()
        elapsed = time.perf_counter() - started

        self.assertLess(
            elapsed, 0.6,
            f"stop_server() 阻塞了主线程 {elapsed:.2f}s；"
            "停止必须在工作线程上执行，否则界面会无响应",
        )
        # The request must have been registered immediately...
        self.assertTrue(self.app._stop_requested)
        self.assertIsNotNone(self.app._stop_thread, "未在工作线程上执行停止")

        # ...and the child must actually die, without the UI having been blocked.
        self.assertTrue(self._pump_until(lambda: not runner.running),
                        "停止请求之后子进程仍在运行")
        self.assertTrue(self._pump_until(lambda: self.app._stop_thread is None),
                        "停止线程未结束")
        self.assertEqual(self.errors, [])

    def test_ui_stays_responsive_while_stopping(self) -> None:
        """Tk must keep processing events during the shutdown."""
        runner = self._start_dummy_child()
        self._slow_down_stop(runner, 1.5)
        self.pump(0.4)

        ticks: list[int] = []
        self.app.stop_server()

        # A blocking stop would prevent these callbacks from ever running.
        handle: dict[str, str] = {}

        def beat() -> None:
            ticks.append(1)
            if runner.running and len(ticks) < 100:
                handle["id"] = self.app.root.after(50, beat)

        def cancel_beat() -> None:
            pending = handle.get("id")
            if pending:
                try:
                    self.app.root.after_cancel(pending)
                except (tk.TclError, ValueError):
                    pass
                handle["id"] = ""

        self.addCleanup(cancel_beat)
        handle["id"] = self.app.root.after(50, beat)

        self.assertTrue(self._pump_until(lambda: not runner.running, timeout=20.0),
                        "子进程未停止")
        cancel_beat()
        self.assertGreater(len(ticks), 2,
                           "停止期间事件循环被阻塞，界面会表现为无响应")

    def test_restart_does_not_block_the_ui_thread(self) -> None:
        runner = self._start_dummy_child()
        self._slow_down_stop(runner, 1.5)
        self.pump(0.4)

        started = time.perf_counter()
        self.app.restart_server()
        elapsed = time.perf_counter() - started

        self.assertLess(
            elapsed, 0.6,
            f"restart_server() 阻塞了主线程 {elapsed:.2f}s；重启必须异步完成",
        )
        # The stop must have been handed to a worker, and must still be running:
        # if restart had waited inline, the child would already be gone here.
        self.assertIsNotNone(self.app._stop_thread, "重启未在工作线程上执行停止")
        self.assertTrue(runner.running, "restart_server() 内联等待了停止完成")

        # And the restart must eventually finish the shutdown.
        self.assertTrue(self._pump_until(lambda: not runner.running, timeout=20.0),
                        "重启过程中子进程未停止")
        self.assertEqual(self.errors, [])

    def test_stop_is_idempotent_while_in_flight(self) -> None:
        """Clicking stop twice must not start two escalation chains."""
        self._shorten_stop_timeouts()
        runner = self._start_dummy_child()
        self.pump(0.4)

        self.app.stop_server()
        first = self.app._stop_thread
        self.app.stop_server()
        self.assertIs(self.app._stop_thread, first, "重复点击启动了第二个停止线程")

        self.assertTrue(self._pump_until(lambda: not runner.running))
        self.assertEqual(self.errors, [])

    def test_stop_when_nothing_is_running_is_a_no_op(self) -> None:
        self.pump(0.3)
        started = time.perf_counter()
        self.app.stop_server()
        self.assertLess(time.perf_counter() - started, 0.3)
        self.assertEqual(self.errors, [])

    def test_stop_callbacks_run_even_when_nothing_is_running(self) -> None:
        """A restart must not hang when the server already exited."""
        self.pump(0.3)
        fired: list[str] = []
        self.app._begin_stop(on_done=lambda: fired.append("done"))
        self.assertEqual(fired, ["done"], "无进程运行时，停止回调未立即执行")

    def test_on_close_calls_back_only_once(self) -> None:
        self.pump(0.3)
        calls: list[int] = []
        original = self.app._finalize_close

        def counting() -> None:
            calls.append(1)
            original()

        self.app._finalize_close = counting  # type: ignore[method-assign]
        self.app.on_close()
        self.app.on_close()
        self.assertEqual(len(calls), 1, "关窗流程执行了多次")

    def test_on_close_without_a_server_is_immediate(self) -> None:
        self.pump(0.3)
        started = time.perf_counter()
        self.app.on_close()
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.0, f"无进程时关窗耗时 {elapsed:.2f}s")
        self.assertTrue(self.app._close_finalized)

    def test_api_tab_shows_endpoints_and_snippets(self) -> None:
        self.pump(0.2)
        self.app.notebook.select(self.app.api_tab)
        self.pump(0.2)
        endpoints = self.app.api_tab.endpoint_text.get("1.0", "end-1c")
        self.assertIn("/v1/chat/completions", endpoints)
        self.assertIn("/health", endpoints)
        snippet = self.app.api_tab.snippet_text.get("1.0", "end-1c")
        self.assertIn("curl", snippet)
        self.assertEqual(self.errors, [])

    def test_preset_round_trip_through_the_ui(self) -> None:
        from llamagui import config as config_mod

        self.pump(0.2)
        preset = config_mod.Preset(
            name="smoke-test",
            values={"--ctx-size": 4096, "--temp": 0.3},
            model="",
        )
        self.app.apply_preset(preset)
        self.pump(0.2)
        self.assertEqual(self.app.get_value("--ctx-size"), 4096)
        built = self.app.build_command()
        self.assertIn("4096", built.argv)
        self.assertEqual(self.errors, [])

    def test_validation_errors_surface_for_a_bad_model(self) -> None:
        self.pump(0.2)
        self.app.set_value("--model", r"C:\no\such\file.gguf")
        built = self.app.build_command()
        self.assertFalse(built.validation.ok)
        self.assertEqual(self.errors, [])

    def test_stop_without_start_is_harmless(self) -> None:
        self.pump(0.2)
        self.app.stop_server()
        self.pump(0.2)
        self.assertFalse(self.app.runner.running)
        self.assertEqual(self.errors, [])

    def test_open_web_ui_url_is_well_formed(self) -> None:
        self.pump(0.1)
        self.app.set_value("--host", "127.0.0.1")
        self.app.set_value("--port", 8899)
        client = self.app.make_client()
        self.assertEqual(client.web_ui_url, "http://127.0.0.1:8899/")
        self.assertEqual(client.base_url, "http://127.0.0.1:8899")
        self.assertEqual(client.openai_base, "http://127.0.0.1:8899/v1")

    def test_api_prefix_and_key_are_reflected_in_the_client(self) -> None:
        self.pump(0.1)
        self.app.set_value("--host", "127.0.0.1")
        self.app.set_value("--port", 8080)
        self.app.set_value("--api-prefix", "llama")
        self.app.set_value("--api-key", "secret1,secret2")
        client = self.app.make_client()
        self.assertEqual(client.base_url, "http://127.0.0.1:8080/llama")
        self.assertEqual(client.api_key, "secret1")
        self.assertIn("/v1/chat/completions", client.url("/v1/chat/completions"))

    def test_ssl_options_switch_the_client_to_https(self) -> None:
        self.pump(0.1)
        self.app.set_value("--ssl-cert-file", "cert.pem")
        client = self.app.make_client()
        self.assertTrue(client.use_ssl)
        self.assertTrue(client.web_ui_url.startswith("https://"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

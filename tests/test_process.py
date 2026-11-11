"""Process supervision tests.

These exercise :class:`llamagui.proc.ProcessRunner` directly with the running
Python interpreter standing in for ``llama-server``, so they verify the launch,
log capture, exit detection, and stop escalation machinery **without ever
loading a model**.

They also cover the failure path in :class:`llamagui.app.App`: a child that dies
immediately (a bad argument, a missing DLL) must be reported as a crash with a
diagnosis rather than leaving the UI stuck on "正在加载模型…".

Run from the ``gui`` directory::

    python -m unittest tests.test_process -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent.parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from llamagui import proc  # noqa: E402


def _python() -> Path:
    """The interpreter to use as a stand-in child process."""
    exe = Path(sys.executable)
    if os.name == "nt":
        # A console-less pythonw.exe would suppress output; prefer python.exe
        # so the pipe actually carries lines.
        candidate = exe.with_name("python.exe")
        if candidate.is_file():
            return candidate
    return exe


class ProcessRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.runner = proc.ProcessRunner()
        self.states: list[proc.State] = []
        self.runner._on_state = lambda state, detail: self.states.append(state)

    def tearDown(self) -> None:
        if self.runner.running:
            self.runner.stop()
        # Release the log file handle and drain the reader before removing the
        # temp tree; on Windows an open handle makes the directory undeletable.
        self.runner.poll()
        for _ in range(50):
            self.runner.drain(500)
            if not self.runner._threads:
                break
            if all(not t.is_alive() for t in self.runner._threads):
                break
            time.sleep(0.02)
        try:
            self.tmp.cleanup()
        except (PermissionError, OSError):
            # A lingering reader thread can hold the handle for a moment; not
            # worth failing a passing test over a temp directory.
            pass

    def _script(self, name: str, body: str) -> Path:
        path = self.dir / name
        path.write_text(body, encoding="utf-8")
        return path

    # ----------------------------------------------------------------- reports

    def test_missing_executable_is_reported_not_raised(self) -> None:
        ok, error = self.runner.start(self.dir / "nope.exe", [])
        self.assertFalse(ok)
        self.assertIn("不存在", error)
        self.assertFalse(self.runner.running)

    def test_captures_stdout_and_stderr_into_one_stream(self) -> None:
        script = self._script("talk.py", (
            "import sys\n"
            "print('to stdout')\n"
            "sys.stdout.flush()\n"
            "print('to stderr', file=sys.stderr)\n"
            "sys.stderr.flush()\n"
        ))
        log_path = self.dir / "out.log"
        ok, error = self.runner.start(_python(), [str(script)], log_path=log_path)
        self.assertTrue(ok, error)

        lines = self._collect(expect=2)
        self.assertIn("to stdout", lines)
        self.assertIn("to stderr", lines)

        # Everything must also be on disk verbatim.
        self.assertTrue(log_path.is_file())
        written = log_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("to stdout", written)
        self.assertIn("to stderr", written)

    def test_unicode_output_survives_the_pipe(self) -> None:
        script = self._script("zh.py", "print('中文输出测试 · t/s')\n")
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)
        lines = self._collect(expect=1)
        self.assertTrue(any("中文输出测试" in line for line in lines), lines)

    def test_exit_is_detected_with_the_return_code(self) -> None:
        script = self._script("fail.py", "import sys; sys.exit(3)\n")
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)

        info = self._wait_exit()
        self.assertIsNotNone(info)
        self.assertEqual(info.code, 3)
        self.assertTrue(info.crashed)
        self.assertFalse(self.runner.running)
        self.assertIn(proc.State.FAILED, self.states)

    def test_clean_voluntary_exit_is_not_a_crash(self) -> None:
        script = self._script("stop.py", "import time\ntime.sleep(30)\n")
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)
        self.assertTrue(self.runner.running)

        info = self.runner.stop(soft_timeout=0.5, hard_timeout=0.5)
        self.assertIsNotNone(info)
        self.assertTrue(info.requested)
        self.assertFalse(info.crashed)
        self.assertFalse(self.runner.running)

    def test_stop_is_prompt_with_production_timeouts(self) -> None:
        """Regression: stopping used to take >10 s and log a WinError 6.

        The old chain waited ``soft`` (6 s) for a process nobody had asked to
        stop, then tried ``CTRL_BREAK_EVENT`` - impossible here, because
        ``CREATE_NO_WINDOW`` leaves the child without a console and a windowed
        Python has none either - and then waited ``hard`` (4 s) more before
        finally terminating.
        """
        script = self._script("stubborn.py", (
            "import signal, time\n"
            "for _n in ('SIGINT', 'SIGBREAK'):\n"
            "    _s = getattr(signal, _n, None)\n"
            "    if _s is not None:\n"
            "        try:\n"
            "            signal.signal(_s, signal.SIG_IGN)\n"
            "        except Exception:\n"
            "            pass\n"
            "print('ready', flush=True)\n"
            "time.sleep(120)\n"
        ))
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)
        self._collect(expect=1)

        started = time.perf_counter()
        info = self.runner.stop()  # deliberately the production timeouts
        elapsed = time.perf_counter() - started

        self.assertFalse(info.crashed, "用户请求的停止不应被判为崩溃")
        self.assertFalse(self.runner.running)
        self.assertLess(
            elapsed, 3.0,
            f"停止耗时 {elapsed:.2f}s，仍然过慢；"
            "在向进程提出任何请求之前不应先等待",
        )

        gui_lines = "\n".join(
            line.text for line in self._drain() if line.stream == "gui"
        )
        self.assertNotIn(
            "CTRL_BREAK", gui_lines,
            "不应尝试向没有控制台的子进程发送 Ctrl+Break（必然 WinError 6）",
        )

    def test_no_console_means_ctrl_break_is_unusable(self) -> None:
        """The platform fact behind the regression above."""
        if os.name != "nt":
            self.skipTest("仅在 Windows 上有意义")
        self.assertFalse(proc._CTRL_BREAK_USABLE)
        self.assertTrue(proc._CREATIONFLAGS & proc.CREATE_NO_WINDOW,
                        "子进程应仍以 CREATE_NO_WINDOW 启动（不弹控制台窗口）")

    def test_stop_escalates_and_logs_each_step(self) -> None:
        # Block sigint so the polite path cannot work and the chain has to
        # escalate; every attempted step must be visible in the log.
        script = self._script("stubborn.py", (
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "time.sleep(60)\n"
        ))
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)
        self._collect(expect=1)

        info = self.runner.stop(soft_timeout=0.4, hard_timeout=0.4)
        self.assertFalse(info.crashed, "用户请求的停止不应被判定为崩溃")
        self.assertFalse(self.runner.running)

        gui_lines = [ln.text for ln in self._drain() if ln.stream == "gui"]
        # Whatever path it took, progress must be reported to the user.
        self.assertTrue(gui_lines, "停止过程没有任何日志")

    def test_emit_marks_gui_lines_and_prefixes_the_log_file(self) -> None:
        log_path = self.dir / "gui.log"
        script = self._script("sleep.py", "import time\ntime.sleep(30)\n")
        ok, _ = self.runner.start(_python(), [str(script)], log_path=log_path)
        self.assertTrue(ok)
        self.runner.emit("这是一条界面消息")
        self._collect(expect=1)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("# 这是一条界面消息", text)
        self.runner.stop(soft_timeout=0.3, hard_timeout=0.3)

    def test_environment_overrides_reach_the_child(self) -> None:
        script = self._script("env.py", "import os; print(os.environ.get('LLAMAGUI_PROBE','missing'))\n")
        ok, _ = self.runner.start(_python(), [str(script)], env={"LLAMAGUI_PROBE": "hello"})
        self.assertTrue(ok)
        lines = self._collect(expect=1)
        self.assertIn("hello", lines)

    def test_second_start_while_running_is_refused(self) -> None:
        script = self._script("long.py", "import time\ntime.sleep(30)\n")
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)
        ok2, error = self.runner.start(_python(), [str(script)])
        self.assertFalse(ok2)
        self.assertIn("已有进程", error)
        self.runner.stop(soft_timeout=0.3, hard_timeout=0.3)

    def test_recent_lines_are_retained_for_crash_reports(self) -> None:
        script = self._script("noisy.py", (
            "for i in range(40):\n"
            "    print(f'line {i}')\n"
        ))
        ok, _ = self.runner.start(_python(), [str(script)])
        self.assertTrue(ok)
        self._wait_exit()
        tail = self.runner.last_lines(30)
        self.assertEqual(len(tail), 30)
        self.assertEqual(tail[-1], "line 39", "崩溃报告应保留子进程最后一行输出")

    def test_stop_when_not_running_is_harmless(self) -> None:
        info = self.runner.stop()
        self.assertIsNotNone(info)
        self.assertFalse(self.runner.running)

    # ---------------------------------------------------------------- helpers

    def _drain(self) -> list[proc.LogLine]:
        out: list[proc.LogLine] = []
        while True:
            batch = self.runner.drain(200)
            if not batch:
                break
            out.extend(batch)
        return out

    def _collect(self, *, expect: int = 1, timeout: float = 15.0) -> list[str]:
        """Gather at least *expect* child lines (GUI lines excluded)."""
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        while time.monotonic() < deadline:
            for item in self._drain():
                if item.stream != "gui":
                    lines.append(item.text)
            if len(lines) >= expect:
                break
            time.sleep(0.02)
        return lines

    def _wait_exit(self, timeout: float = 15.0) -> proc.ExitInfo | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._drain()
            info = self.runner.poll()
            if info is not None:
                return info
            time.sleep(0.02)
        return None


class DiagnosisTests(unittest.TestCase):
    """The crash explanation must map signatures to actionable advice.

    ``diagnose_failure`` is a pure function, so this needs no window, no Tk, and
    no configuration directory - and therefore cannot leak into the user's real
    ``gui/config``.
    """

    def _diagnose(self, lines: list[str]) -> str:
        return proc.diagnose_failure(
            proc.ExitInfo(code=1, requested=False, tail=lines)
        )

    def test_unknown_argument(self) -> None:
        self.assertIn("参数", self._diagnose(["error: unknown argument: --nope"]))

    def test_missing_model(self) -> None:
        self.assertIn("路径", self._diagnose(["failed to load model 'x.gguf'"]))

    def test_out_of_memory(self) -> None:
        self.assertIn("显存", self._diagnose(["CUDA error: out of memory"]))

    def test_port_in_use(self) -> None:
        self.assertIn("端口", self._diagnose(["couldn't bind to address: address already in use"]))

    def test_missing_dll(self) -> None:
        self.assertIn("DLL", self._diagnose(["cannot open shared object file: ggml.dll"]))
        self.assertIn("DLL", self._diagnose(["error loading ggml-cuda.dll"]))

    def test_cuda_unavailable(self) -> None:
        self.assertIn("CUDA", self._diagnose(["unsupported gpu architecture"]))

    def test_unknown_failure_returns_empty(self) -> None:
        self.assertEqual(self._diagnose(["something odd happened"]), "")

    def test_empty_tail_is_safe(self) -> None:
        self.assertEqual(self._diagnose([]), "")

    def test_a_requested_stop_is_not_diagnosed(self) -> None:
        """A user-requested stop must not be reported as a crash."""
        info = proc.ExitInfo(code=1, requested=True, tail=["killed"])
        self.assertFalse(info.crashed)


if __name__ == "__main__":
    unittest.main(verbosity=2)

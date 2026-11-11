"""Supervision of the ``llama-server`` child process.

Design notes that matter for stability:

* The child is created in its own process group with no console window, so the
  GUI never flashes a black box and can send ``CTRL_BREAK_EVENT`` to request a
  graceful shutdown.
* Output is drained by dedicated reader threads into a bounded queue, so a
  chatty startup log can never block the child's pipe (which would stall the
  server) and can never grow without bound.
* Shutdown escalates: polite, then Ctrl+Break, then terminate, then
  ``taskkill /T /F``.  Each step is logged so a user can see how far it went.
* The full raw log is also written to ``config/logs`` so nothing is lost even
  if the UI truncates its display.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- platform bits

if os.name == "nt":  # pragma: no cover - platform dependent
    CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    _CREATIONFLAGS = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    CREATIONFLAGS_NO_WINDOW = CREATE_NO_WINDOW
else:
    _CREATIONFLAGS = 0
    CREATIONFLAGS_NO_WINDOW = 0

#: Whether a console control event (Ctrl+Break) can reach the child.
#:
#: It cannot, in this launcher's configuration, for two independent reasons:
#:
#: * ``CREATE_NO_WINDOW`` runs the child *without a console at all*, so it is not
#:   attached to any console that could deliver such an event;
#: * a GUI started through ``pythonw.exe`` has no console of its own either, and
#:   ``GenerateConsoleCtrlEvent`` needs one on the caller's side too.
#:
#: The attempt therefore fails with ``OSError: [WinError 6] 句柄无效``
#: (ERROR_INVALID_HANDLE) - a structural limitation, not a transient glitch.
#: Making it work would mean giving the child a console, i.e. a console window,
#: which is precisely what this launcher exists to avoid.  Deriving the flag
#: from the creation flags keeps the step dormant instead of calling something
#: that always fails and then waiting out its timeout for nothing.
_CTRL_BREAK_USABLE = (
    os.name == "nt"
    and hasattr(signal, "CTRL_BREAK_EVENT")
    and bool(_CREATIONFLAGS & CREATE_NEW_PROCESS_GROUP)
    and not bool(_CREATIONFLAGS & CREATE_NO_WINDOW)
)

#: Line queue capacity.  Dropping the oldest line keeps the pipe flowing.
_LOG_QUEUE_MAX = 20000

#: How long :meth:`ProcessRunner.stop` waits at each escalation step before
#: moving to the next one.  Module level so the behaviour is tunable and so
#: tests can shorten it; the totals bound how long a shutdown can take.
DEFAULT_SOFT_TIMEOUT = 6.0
DEFAULT_HARD_TIMEOUT = 4.0
DEFAULT_TERMINATE_TIMEOUT = 2.0


class State(str, Enum):
    """Lifecycle state of the supervised server."""

    IDLE = "idle"
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    STOPPING = "stopping"
    EXITED = "exited"
    FAILED = "failed"


@dataclass
class LogLine:
    """One line of child output."""

    text: str
    stream: str = "stdout"  # stdout | stderr | gui
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExitInfo:
    """How the process ended."""

    code: Optional[int] = None
    requested: bool = False
    duration_s: float = 0.0
    tail: List[str] = field(default_factory=list)

    @property
    def crashed(self) -> bool:
        return not self.requested and self.code not in (0, None)


class ProcessRunner:
    """Owns one ``llama-server`` process at a time."""

    def __init__(
        self,
        *,
        queue_max: int = _LOG_QUEUE_MAX,
        on_state: Optional[Callable[[State, object], None]] = None,
    ) -> None:
        self.lines: "queue.Queue[LogLine]" = queue.Queue(maxsize=queue_max)
        self.events: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.state: State = State.IDLE
        self._proc: Optional[subprocess.Popen] = None
        self._threads: List[threading.Thread] = []
        self._log_handle = None
        self._log_path: Optional[Path] = None
        self._started_at: float = 0.0
        self._requested_stop = False
        self._recent: List[str] = []
        self._recent_cap = 200
        self._lock = threading.Lock()
        self._on_state = on_state
        self.argv: List[str] = []
        self.env: Dict[str, str] = {}
        self.cwd: Optional[Path] = None
        #: Child output captured at exit, so a crash report shows the real tail.
        self._tail_snapshot: Optional[List[str]] = None
        #: Set once the process has been reaped, making _finish idempotent.
        self._last_exit: Optional[ExitInfo] = None

    # ------------------------------------------------------------------ status

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        with self._lock:
            return self._proc.pid if self._proc is not None else None

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_path

    @property
    def uptime_s(self) -> float:
        return time.time() - self._started_at if self._started_at else 0.0

    def last_lines(self, count: int = 30) -> List[str]:
        """The most recent child output lines.

        After the process exits this returns the snapshot taken at exit, not the
        live ring buffer: the runner appends its own "exited with code N" note
        right after finishing, and that note must not displace the child's real
        final lines in a crash report.
        """
        with self._lock:
            if self._proc is None and self._tail_snapshot is not None:
                return list(self._tail_snapshot[-count:])
            return list(self._recent[-count:])

    def set_state(self, state: State, detail: object = None) -> None:
        self.state = state
        if self._on_state is not None:
            try:
                self._on_state(state, detail)
            except Exception:  # noqa: BLE001 - a UI callback must never kill us
                pass
        self.events.put(("state", (state, detail)))

    def emit(self, text: str, stream: str = "gui") -> None:
        """Push a GUI-generated line into the same stream as child output."""
        self._push(LogLine(text=text, stream=stream))

    # ------------------------------------------------------------------- start

    def start(
        self,
        executable: Path,
        argv: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[Path] = None,
    ) -> Tuple[bool, str]:
        """Launch the server.  Returns ``(ok, error_message)``."""
        if self.running:
            return False, "已有进程在运行，请先停止。"

        exe = Path(executable)
        if not exe.is_file():
            return False, f"可执行文件不存在：{exe}"

        merged = os.environ.copy()
        for key, value in (env or {}).items():
            if key:
                merged[str(key)] = str(value)
        # Keep the GUI's own choice of colour handling predictable in a pipe.
        merged.setdefault("PYTHONIOENCODING", "utf-8")

        self.argv = [str(exe), *[str(a) for a in argv]]
        self.env = dict(env or {})
        self.cwd = Path(cwd) if cwd else exe.parent
        self._recent = []
        self._tail_snapshot = None
        self._last_exit = None
        self._requested_stop = False

        if log_path is not None:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
                self._log_path = Path(log_path)
            except OSError:
                self._log_handle = None
                self._log_path = None

        self.set_state(State.STARTING)
        try:
            proc = subprocess.Popen(
                self.argv,
                cwd=str(self.cwd),
                env=merged,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_CREATIONFLAGS,
            )
        except OSError as exc:
            self._close_log()
            self.set_state(State.FAILED, str(exc))
            return False, f"启动失败：{exc}"

        with self._lock:
            self._proc = proc
        self._started_at = time.time()
        self.emit(f"$ {' '.join(self.argv)}")
        self.emit(f"[gui] 工作目录 {self.cwd}")
        if env:
            for key, value in env.items():
                shown = value if len(str(value)) < 60 else f"{str(value)[:57]}..."
                self.emit(f"[gui] 环境变量 {key}={shown}")

        reader = threading.Thread(
            target=self._reader, args=(proc,), name="llamagui-reader", daemon=True
        )
        reader.start()
        self._threads = [reader]
        return True, ""

    def _reader(self, proc: subprocess.Popen) -> None:
        """Drain the merged output at line granularity."""
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self._push(LogLine(text=line.rstrip("\n"), stream="stdout"))
        except (ValueError, OSError):
            # The pipe closes when the process exits; that is expected.
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _push(self, line: LogLine) -> None:
        with self._lock:
            self._recent.append(line.text)
            if len(self._recent) > self._recent_cap:
                del self._recent[: len(self._recent) - self._recent_cap]
            handle = self._log_handle
        if handle is not None:
            try:
                prefix = "" if line.stream != "gui" else "# "
                handle.write(f"{prefix}{line.text}\n")
                handle.flush()
            except (OSError, ValueError):
                pass
        try:
            self.lines.put_nowait(line)
        except queue.Full:
            # Prefer keeping the newest output: drop one old line and retry once.
            try:
                self.lines.get_nowait()
            except queue.Empty:
                pass
            try:
                self.lines.put_nowait(line)
            except queue.Full:
                pass

    # -------------------------------------------------------------------- stop

    def stop(
        self,
        *,
        graceful: Optional[Callable[[], bool]] = None,
        soft_timeout: Optional[float] = None,
        hard_timeout: Optional[float] = None,
        terminate_timeout: Optional[float] = None,
    ) -> ExitInfo:
        """Stop the child, escalating as needed.

        The chain is deliberately short, because every wait that is not preceded
        by an actual request to the process is pure dead time:

        1. *graceful*, when the caller supplies a working mechanism, then wait
           ``soft_timeout``.  No such mechanism exists for this launcher, so by
           default this step does not run at all.
        2. ``CTRL_BREAK_EVENT``, only when :data:`_CTRL_BREAK_USABLE` - i.e. when
           the child actually has a console.  Then wait ``hard_timeout``.
        3. ``terminate()`` (TerminateProcess), which is immediate and always
           works, then wait ``terminate_timeout``.
        4. ``taskkill /T /F`` as a last resort.

        Callers on a UI thread must run this on a worker: even the short path
        performs a few blocking polls.
        """
        soft = DEFAULT_SOFT_TIMEOUT if soft_timeout is None else soft_timeout
        hard = DEFAULT_HARD_TIMEOUT if hard_timeout is None else hard_timeout
        terminate = (
            DEFAULT_TERMINATE_TIMEOUT if terminate_timeout is None else terminate_timeout
        )

        if not self.running:
            self.set_state(State.IDLE)
            return ExitInfo(code=None, requested=True)

        self._requested_stop = True
        self.set_state(State.STOPPING)
        started = time.time()

        # --- 1) polite, only if we actually asked for something --------------
        if graceful is not None:
            asked = False
            try:
                asked = bool(graceful())
            except Exception as exc:  # noqa: BLE001
                self.emit(f"[gui] 优雅退出请求失败：{exc}")
            if asked:
                self.emit("[gui] 已请求服务端优雅退出，等待其自行结束…")
                if self._wait_exit(soft):
                    return self._finish(started)
                self.emit("[gui] 服务端未在等待时间内结束，改为强制结束")

        # --- 2) console control event, only when it can possibly work -------
        if _CTRL_BREAK_USABLE:
            with self._lock:
                proc = self._proc
            if proc is not None:
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    self.emit("[gui] 已发送 CTRL_BREAK_EVENT，等待退出…")
                    if self._wait_exit(hard):
                        return self._finish(started)
                except (OSError, ValueError) as exc:
                    self.emit(f"[gui] 发送 CTRL_BREAK_EVENT 失败：{exc}")

        # --- 3) the mechanism that actually works here ----------------------
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            self.emit("[gui] 正在结束进程…")
            try:
                proc.terminate()
            except OSError as exc:
                self.emit(f"[gui] terminate() 失败：{exc}")
            if self._wait_exit(terminate):
                return self._finish(started)

        # --- 4) last resort -------------------------------------------------
        self._taskkill_tree()
        self._wait_exit(3.0)
        return self._finish(started)

    def _wait_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            with self._lock:
                proc = self._proc
            if proc is None or proc.poll() is not None:
                return True
            time.sleep(0.08)
        with self._lock:
            proc = self._proc
        return proc is None or proc.poll() is not None

    def _taskkill_tree(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.emit(f"[gui] 强制结束进程树 PID {proc.pid}")
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.emit(f"[gui] taskkill 失败：{exc}")

    def _finish(self, started: float) -> ExitInfo:
        """Record the exit exactly once, whoever notices it first.

        Both the UI's periodic :meth:`poll` and an in-flight :meth:`stop` can
        observe the process ending.  Without this guard the second caller would
        re-snapshot the log tail (now including our own "exited with code N"
        note) and lose the real exit code.
        """
        with self._lock:
            if self._proc is None:
                # Already reaped (or a concurrent caller is reaping it).
                return self._last_exit or ExitInfo(code=None, requested=True)
            proc = self._proc
        # The child has exited, but its last lines - usually the actual error
        # message of a crash - may still be in the pipe.  Let the reader drain
        # them before taking the snapshot.  EOF arrives as soon as the pipe is
        # empty, so this normally returns at once; the timeout only matters if
        # a grandchild inherited the pipe and keeps it open.
        current = threading.current_thread()
        for thread in self._threads:
            if thread is not current and thread.is_alive():
                thread.join(timeout=1.0)
        with self._lock:
            if self._proc is not proc:
                return self._last_exit or ExitInfo(code=None, requested=True)
            self._proc = None
            # Snapshot the child's own output before anything else can append.
            self._tail_snapshot = list(self._recent[-30:])
            code = proc.returncode
            info = ExitInfo(
                code=code,
                requested=self._requested_stop,
                # How long the server ran, not how long stopping it took.
                duration_s=time.time() - (self._started_at or started),
                tail=list(self._tail_snapshot),
            )
            self._last_exit = info
        self._close_log()
        self._started_at = 0.0
        if info.crashed:
            self.set_state(State.FAILED, info)
        else:
            self.set_state(State.IDLE)
        return info

    # ------------------------------------------------------------------- poll

    def poll(self) -> Optional[ExitInfo]:
        """Detect an unexpected exit.  Call periodically from the UI loop."""
        with self._lock:
            proc = self._proc
        if proc is None:
            return None
        code = proc.poll()
        if code is None:
            return None
        # Finish first so the crash tail is exactly the child's own output: an
        # "exited with code N" note appended by us would otherwise displace the
        # last real line and be fairly useless in a crash report.
        info = self._finish(self._started_at or time.time())
        if info is not None and not info.requested:
            self.emit(f"[gui] 进程已退出，返回码 {code}")
        return info

    def _close_log(self) -> None:
        with self._lock:
            handle = self._log_handle
            self._log_handle = None
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------ output

    def drain(self, limit: int = 400) -> List[LogLine]:
        """Take up to *limit* queued lines without blocking."""
        out: List[LogLine] = []
        for _ in range(limit):
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return out


# ------------------------------------------------------------------- utilities


def list_devices(executable: Path, timeout: float = 25.0) -> Tuple[List[str], str]:
    """Run ``llama-bench --list-devices`` and return its device lines.

    ``llama-bench`` is used rather than ``llama-server`` because it exits
    immediately after enumerating, so no port is bound and no model is touched.
    Returns an empty list plus an error string on failure; the GUI treats device
    enumeration as best-effort.
    """
    exe = Path(executable)
    if not exe.is_file():
        return [], f"未找到 {exe.name}"
    try:
        proc = subprocess.run(
            [str(exe), "--list-devices"],
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATIONFLAGS_NO_WINDOW,
            cwd=str(exe.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"枚举设备失败：{exc}"

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    devices: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Device ") or stripped.startswith("Available devices"):
            continue
        # Real entries look like "CUDA0: NVIDIA GeForce RTX 3080 (20479 MiB, ...)"
        if ":" in stripped and any(
            stripped.startswith(prefix)
            for prefix in ("CUDA", "CPU", "ROCm", "Vulkan", "SYCL", "Metal", "RPC")
        ):
            devices.append(stripped)
    return devices, "" if devices else proc.stderr.strip()[:400]


def probe_version(executable: Path, timeout: float = 15.0) -> Tuple[str, str]:
    """Return ``(version_banner, error)`` for a llama.cpp executable."""
    exe = Path(executable)
    if not exe.is_file():
        return "", f"未找到 {exe.name}"
    try:
        proc = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATIONFLAGS_NO_WINDOW,
            cwd=str(exe.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"读取版本失败：{exc}"
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    for line in output.splitlines():
        if line.strip().lower().startswith("version:"):
            return line.split(":", 1)[1].strip(), ""
    return output.strip(), ""


def capture_help(executable: Path, timeout: float = 30.0) -> Tuple[str, str]:
    """Return ``(help_text, error)`` from an executable's ``--help``."""
    exe = Path(executable)
    if not exe.is_file():
        return "", f"未找到 {exe.name}"
    try:
        proc = subprocess.run(
            [str(exe), "--help"],
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATIONFLAGS_NO_WINDOW,
            cwd=str(exe.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"读取帮助失败：{exc}"
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return text, "" if text.strip() else "帮助输出为空"


def diagnose_failure(info: "ExitInfo") -> str:
    """Map a crashed process's output to actionable advice.

    Pure: reads only ``info.tail`` and returns a string.  It lives here rather
    than on the application object so it can be unit tested without constructing
    a window or touching the user's configuration.
    """
    tail = "\n".join(info.tail).lower()
    if "unknown argument" in tail or "invalid argument" in tail:
        return "命令行参数不被支持。请检查命令详情，或点击「运行 → 刷新参数表」。"
    if "failed to load model" in tail or "no such file" in tail:
        return "模型路径错误或文件损坏。"
    if ("out of memory" in tail or "cuda error" in tail
            or ("alloc" in tail and "fail" in tail)):
        return ("显存或内存不足。可降低 -c、改用 q8_0 KV、调小 -ub，"
                "或清空 -ngl 让 --fit 自动决定。")
    if "address already in use" in tail or "couldn't bind" in tail:
        return "端口被占用。换一个端口后重试。"
    if "cannot open shared object" in tail or ".dll" in tail or "error loading" in tail:
        return "缺少运行库 DLL，请确认 llama.cpp 目录完整。"
    if "unsupported gpu" in tail or ("cuda" in tail and "not" in tail):
        return "CUDA 不可用。可设置 -dev none 改用 CPU。"
    return ""

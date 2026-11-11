"""The main window and application state machine.

Threading model, which is the part that matters for stability:

* The Tk main thread owns all widgets and the config.  It never blocks.
* Long operations (GGUF parsing, device enumeration, HTTP polling, waiting for
  readiness) run on worker threads and hand results back through
  ``self._ui_queue``, drained by a timer on the main thread.
* Child process output is drained by :mod:`llamagui.proc`'s reader thread into a
  queue, which the same timer consumes.
* Every command-line build happens on the main thread with the config already
  flushed, so what the preview shows is exactly what gets launched.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from . import argspec, cmdline, config as config_mod, gguf, helpparse, httpclient, logparse, paths
from .argspec import ArgSpec, Group
from .proc import ExitInfo, ProcessRunner, State, diagnose_failure
from .ui.tabs import (
    ApiTab,
    BaseTab,
    BenchTab,
    ConsoleTab,
    HardwareTab,
    ModelTab,
    ParamTab,
    TabHost,
)

#: UI timer intervals in milliseconds.
_TICK_MS = 120
_HEALTH_INTERVAL_MS = 800
_STATUS_CLEAR_MS = 6000
#: Hard bound on quitting.  Exceeds the worst case of the stop chain
#: (6 s polite + 4 s Ctrl+Break + 2 s terminate + 3 s taskkill) so the window is
#: only destroyed early if a thread has genuinely wedged.
_CLOSE_LIMIT_MS = 20000
#: How many parsed log events to retain for the console's filter/rebuild.
_LOG_HISTORY = 4000

_WINDOW_TITLE = "llama.cpp 启动器"


class App(TabHost):
    """Owns the window, the config, the child process, and the timers."""

    def __init__(self, layout: Optional[paths.Layout] = None) -> None:
        self.layout = layout or paths.layout()
        self.layout.ensure_dirs()
        self.store = config_mod.Store(self.layout)
        self.config = self.store.load()

        self.runner = ProcessRunner(on_state=self._on_runner_state)
        self.parser = logparse.LogParser()
        self.log_history: List[logparse.LogEvent] = []

        #: Worker -> UI messages.  Each item is a callable to run on the Tk thread.
        self._ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._waiting_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        #: ``(host, port, ssl)`` from the running server's "listening on" line.
        self._listen_addr: Optional[Tuple[str, int, bool]] = None
        self._gguf_info: Optional[gguf.GGUFInfo] = None
        self._devices: List[str] = []
        self._pending: Dict[str, Any] = {}
        self._status_after: Optional[str] = None
        self._estimate_after: Optional[str] = None
        self._tick_after: Optional[str] = None
        self._running = False
        #: Worker thread performing the shutdown escalation, if any.
        self._stop_thread: Optional[threading.Thread] = None
        #: Callbacks to run once the child process is gone.
        self._stop_callbacks: List[Callable[[], None]] = []
        self._closing = False
        self._close_finalized = False
        # Declared up front so any code path can read them even if help parsing
        # fails part way through.
        self.build_info: str = ""
        self.help_doc: Optional[helpparse.HelpDoc] = None
        self.help_error: str = ""

        # Order matters: widgets first, then help parsing (which reports through
        # the console tab), then the pre-flight summary and the timers.
        self._build_window()
        self._load_help()
        self._start_timers()
        self._sync_from_help()
        self._preflight_report()

    # ----------------------------------------------------------------- window

    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"{_WINDOW_TITLE} {__version__}")
        self.root.minsize(1080, 660)
        geometry = self.config.window.get("geometry")
        if isinstance(geometry, str) and geometry:
            try:
                self.root.geometry(geometry)
            except tk.TclError:
                self.root.geometry("1280x820")
        else:
            self.root.geometry("1280x820")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._configure_styles()

        self._build_menu()
        self._build_toolbar()
        self._build_notebook()
        self._build_statusbar()

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        # 'vista'/'xpnative' look native on Windows but ignore background
        # colours; 'clam' is chosen so status colours actually render.
        if "clam" in style.theme_names():
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
        style.configure("Status.TLabel", padding=(8, 4))
        style.configure("State.TLabel", font=("Segoe UI", 10, "bold"), padding=(8, 4))
        style.configure("Command.TLabel", font=("Consolas", 9), foreground="#005a9e")
        style.configure("Hint.TLabel", foreground="#666666")
        self.state_colors = {
            "idle": "#555555",
            "starting": "#0b6fb8",
            "loading": "#b06f00",
            "ready": "#0a7a3d",
            "stopping": "#b06f00",
            "failed": "#b00020",
        }

    # -------------------------------------------------------------------- menu

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="保存配置", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="另存为预设…", command=self.save_preset)
        file_menu.add_command(label="载入预设…", command=self.load_preset_dialog)
        file_menu.add_command(label="导入预设文件…", command=self.import_preset)
        file_menu.add_command(label="导出当前配置为预设…", command=self.export_preset)
        file_menu.add_separator()
        file_menu.add_command(label="恢复当前页默认值", command=self.reset_current_tab)
        file_menu.add_command(label="全部重置为默认（清空所有参数）", command=self.reset_all)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="启动", command=self.start_server)
        run_menu.add_command(label="停止", command=self.stop_server)
        run_menu.add_command(label="重启", command=self.restart_server)
        run_menu.add_separator()
        run_menu.add_command(label="打开内置 Web UI", command=self.open_web_ui)
        run_menu.add_command(label="复制命令", command=self.copy_command)
        run_menu.add_separator()
        run_menu.add_command(label="刷新参数表（重新读取 --help）", command=self.refresh_help)
        run_menu.add_command(label="枚举设备", command=self.enumerate_devices)
        menubar.add_cascade(label="运行", menu=run_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于与已知限制", command=self.show_about)
        help_menu.add_command(label="打开配置目录", command=self.open_config_dir)
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.root.configure(menu=menubar)

    # ----------------------------------------------------------------- toolbar

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(5, weight=1)

        self.start_button = ttk.Button(bar, text="▶ 启动", command=self.start_server, width=10)
        self.start_button.grid(row=0, column=0, padx=(0, 4))
        self.stop_button = ttk.Button(bar, text="■ 停止", command=self.stop_server, width=10)
        self.stop_button.grid(row=0, column=1, padx=4)
        ttk.Button(bar, text="⟳ 重启", command=self.restart_server, width=10).grid(row=0, column=2, padx=4)

        self.state_label = ttk.Label(bar, text="空闲", style="State.TLabel")
        self.state_label.grid(row=0, column=3, padx=(12, 4))

        ttk.Button(bar, text="打开 Web UI", command=self.open_web_ui).grid(row=0, column=4, padx=4)

        self.perf_label = ttk.Label(bar, text="", style="Hint.TLabel")
        self.perf_label.grid(row=0, column=5, sticky="e", padx=8)

        self.reset_tab_button = ttk.Button(bar, text="↺ 恢复本页默认",
                                           command=self.reset_current_tab)
        self.reset_tab_button.grid(row=0, column=6, padx=(4, 0))

        preview = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        preview.grid(row=1, column=0, sticky="ew")
        preview.columnconfigure(1, weight=1)
        ttk.Label(preview, text="命令预览", style="Hint.TLabel").grid(row=0, column=0, sticky="w")
        self.command_var = tk.StringVar(value="")
        self.command_label = ttk.Label(preview, textvariable=self.command_var,
                                       style="Command.TLabel", anchor="w")
        self.command_label.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(preview, text="复制", width=6, command=self.copy_command).grid(row=0, column=2)
        ttk.Button(preview, text="详情", width=6, command=self.show_command_dialog).grid(
            row=0, column=3, padx=4
        )

    # ---------------------------------------------------------------- notebook

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 4))

        self.tabs: List[BaseTab] = []
        self.model_tab = ModelTab(self.notebook, self)
        self.hardware_tab = HardwareTab(self.notebook, self)
        self.context_tab = ParamTab(self.notebook, self, "上下文与线程",
                                    [Group.CONTEXT])
        self.sampling_tab = ParamTab(self.notebook, self, "采样与生成",
                                     [Group.SAMPLING])
        self.spec_tab = ParamTab(self.notebook, self, "投机解码",
                                 [Group.SPECULATIVE])
        self.server_tab = ParamTab(self.notebook, self, "服务器与 API",
                                   [Group.SERVER, Group.LOGGING])
        self.console_tab = ConsoleTab(self.notebook, self)
        self.api_tab = ApiTab(self.notebook, self)
        self.bench_tab = BenchTab(self.notebook, self)

        for tab in (self.model_tab, self.hardware_tab, self.context_tab,
                    self.sampling_tab, self.spec_tab, self.server_tab,
                    self.console_tab, self.api_tab, self.bench_tab):
            self.notebook.add(tab, text=tab.title)
            self.tabs.append(tab)

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(0, 0))
        bar.grid(row=3, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.facts_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.facts_var, style="Status.TLabel").grid(
            row=0, column=1, sticky="e"
        )

    # ------------------------------------------------------------------ startup

    def _load_help(self) -> None:
        """Parse the local ``--help`` output, using the cache when possible."""
        self.help_doc: Optional[helpparse.HelpDoc] = None
        self.help_error = ""
        server = self.layout.binary("llama-server.exe")

        # The caches describe one particular build.  When the binaries change
        # (llama.cpp was upgraded) they must be ignored, or the GUI would keep
        # sending flags the new build removed and greying out ones it added.
        fingerprint = binary_fingerprint(self.layout)
        cached_meta = self.store.read_cache_json("fingerprint.json")
        stale = not (isinstance(cached_meta, dict)
                     and cached_meta.get("fingerprint") == fingerprint)

        fallback: Optional[helpparse.HelpDoc] = None
        cached = self.store.read_cache("help_server.txt")
        if cached:
            fallback = helpparse.parse_help(cached, str(server))
        elif self.layout.help_override and self.layout.help_override.is_file():
            try:
                text = self.layout.help_override.read_text(encoding="utf-8", errors="replace")
                fallback = helpparse.parse_help(text, str(self.layout.help_override))
            except OSError:
                fallback = None
        if not stale and fallback is not None and len(fallback.options) >= 50:
            self.help_doc = fallback

        if self.help_doc is None:
            # First run, or the binaries changed: ask the binary itself.  This
            # never loads a model.
            from .proc import capture_help

            text, error = capture_help(server)
            if error:
                self.help_error = error
                # Better an old table than none at all; the error is reported.
                self.help_doc = fallback
            elif text:
                self.help_doc = helpparse.parse_help(text, str(server))
                self.store.write_cache("help_server.txt", text)
                self.store.write_cache_json("fingerprint.json", {"fingerprint": fingerprint})

        banner = ""
        if self.help_doc is not None:
            banner, _ = _server_version(self.layout, fingerprint)
        self._set_build_info(banner)

    def _set_build_info(self, banner: str) -> None:
        self.build_info = banner or ""
        number = helpparse.build_number(banner) if banner else ""
        title = f"{_WINDOW_TITLE} {__version__}"
        if number:
            title += f"  ·  llama.cpp build {number}"
        self.root.title(title)

    def _sync_from_help(self) -> None:
        """Report table entries this build does not support."""
        if self.help_doc is None:
            return
        missing = self.help_doc.missing(argspec.SPECS)
        if missing:
            self.status(f"注意：本 build 未提供 {len(missing)} 个已配置参数，它们不会出现在命令行中")

    def _preflight_report(self) -> None:
        """Show fatal layout problems before the user tries to launch."""
        missing = self.layout.missing_required
        problems: List[str] = []
        if missing:
            problems.append("缺少必需文件：" + "、".join(missing))
        if self.help_error:
            problems.append(f"读取 --help 失败：{self.help_error}")
        if not self.layout.has_cuda_libs:
            problems.append("未找到 CUDA 运行库，GPU 卸载可能不可用（可改用 -dev none 走 CPU）")
        if problems:
            self.status(" · ".join(problems))
            self.console_tab.append("【启动前检查】")
            for problem in problems:
                self.console_tab.append(f"  ⚠ {problem}")
        else:
            self.console_tab.append(
                f"【启动前检查】通过：二进制位于 {self.layout.bin_dir}（{self.layout.source}）"
            )
        if self.help_doc is not None:
            self.console_tab.append(
                f"【参数表】已解析 {len(self.help_doc.options)} 个参数，"
                f"界面暴露 {len(argspec.SPECS)} 个"
            )

    def _start_timers(self) -> None:
        self._running = True
        self._tick_after = self.root.after(_TICK_MS, self._tick)
        self._estimate_after = self.root.after(50, self._do_update_estimate)

    def shutdown(self) -> None:
        """Cancel pending callbacks and stop the child process.

        Called before the window is destroyed.  Without this, ``after()``
        callbacks already queued in Tcl fire against a dead interpreter and Tk
        reports "invalid command name" for each one.

        The tick reschedules itself, so the handle must be tracked and the
        ``_running`` flag cleared first: a tick that is mid-flight when
        ``shutdown`` runs would otherwise queue the next one after the cancel.
        """
        self._running = False
        for attribute in ("_tick_after", "_status_after", "_estimate_after"):
            handle = getattr(self, attribute, None)
            if handle is not None:
                try:
                    self.root.after_cancel(handle)
                except (tk.TclError, ValueError):
                    pass
                setattr(self, attribute, None)
        # Only stop the child here if nobody else is already doing it: a
        # concurrent stop from the close path (or a worker thread) would run the
        # escalation chain twice against the same process.
        thread = self._stop_thread
        if self.runner.running and not (thread is not None and thread.is_alive()):
            self.runner.stop()

    def _cancel_after(self, handle: Optional[str], attribute: str) -> None:
        if handle is not None:
            try:
                self.root.after_cancel(handle)
            except (tk.TclError, ValueError):
                pass
        setattr(self, attribute, None)

    # ------------------------------------------------------------------- timers

    def _tick(self) -> None:
        """Single periodic pump: worker messages, process output, exit detection."""
        self._tick_after = None
        if not self._running:
            return
        self._drain_ui_queue()
        self._drain_logs()
        self._poll_process()
        self._tick_after = self.root.after(_TICK_MS, self._tick)

    def _drain_ui_queue(self) -> None:
        for _ in range(50):
            try:
                job = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                job()
            except tk.TclError:
                pass
            except Exception as exc:  # noqa: BLE001 - a UI job must never kill the pump
                self.console_tab.append(f"[gui] 界面任务异常：{exc!r}")

    def post(self, job: Callable[[], None]) -> None:
        """Queue *job* to run on the Tk main thread."""
        self._ui_queue.put(job)

    def _drain_logs(self) -> None:
        lines = self.runner.drain(400)
        if not lines:
            return
        for line in lines:
            event = self.parser.parse_line(line.text)
            if event is None:
                continue
            if line.stream == "gui":
                event.level = "gui"
            self.log_history.append(event)
            if len(self.log_history) > _LOG_HISTORY:
                del self.log_history[: len(self.log_history) - _LOG_HISTORY]
            self.console_tab.append_parsed(event)
        self._note_listen_url()
        self._refresh_facts()

    def _note_listen_url(self) -> None:
        url = self.parser.facts.listen_url
        if not url:
            return
        addr = httpclient.parse_listen_url(url)
        if addr is None or addr == self._listen_addr:
            return
        self._listen_addr = addr
        expected = self.effective_port()
        if addr[1] != expected:
            self.console_tab.append(
                f"[gui] 服务实际监听 {url}（与预期端口 {expected} 不同），已改用实际地址"
            )
        self.api_tab.update_client()

    def _poll_process(self) -> None:
        info = self.runner.poll()
        if info is not None:
            self._on_process_exit(info)

    def _refresh_facts(self) -> None:
        facts = self.parser.facts
        self.console_tab.set_facts(facts.summary())
        self.facts_var.set(facts.summary())
        if facts.eval_per_second or facts.prompt_per_second:
            bits = []
            if facts.prompt_per_second:
                bits.append(f"预填充 {facts.prompt_per_second:.1f} t/s")
            if facts.eval_per_second:
                bits.append(f"生成 {facts.eval_per_second:.1f} t/s")
            self.perf_label.configure(text=" · ".join(bits))

    # ------------------------------------------------------------- state machine

    def _on_runner_state(self, state: State, detail: Any) -> None:
        self.post(lambda: self._apply_state(state, detail))

    def _apply_state(self, state: State, detail: Any) -> None:
        text = {
            State.IDLE: ("空闲", "idle"),
            State.STARTING: ("启动中…", "starting"),
            State.LOADING: ("正在加载模型…", "loading"),
            State.READY: ("运行中", "ready"),
            State.STOPPING: ("正在停止…", "stopping"),
            State.EXITED: ("已退出", "idle"),
            State.FAILED: ("启动失败", "failed"),
        }[state]
        self.state_label.configure(text=text[0], foreground=self.state_colors[text[1]])
        self.start_button.configure(state="disabled" if state in (
            State.STARTING, State.LOADING, State.READY, State.STOPPING) else "normal")
        self.stop_button.configure(state="normal" if state in (
            State.STARTING, State.LOADING, State.READY) else "disabled")
        self.api_tab.update_client()

    def _on_process_exit(self, info: ExitInfo) -> None:
        # The health-watch thread exits by itself: it checks runner.running.
        self._waiting_thread = None
        if info.crashed:
            self.console_tab.append("")
            self.console_tab.append("【进程异常退出】以下为最后 25 行输出：")
            for line in info.tail[-25:]:
                self.console_tab.append(f"  {line}")
            hint = diagnose_failure(info)
            if hint:
                self.console_tab.append(f"【可能原因】{hint}")
            self.status(f"进程异常退出（返回码 {info.code}）")
            self.state_label.configure(text="启动失败", foreground=self.state_colors["failed"])
        else:
            self.status(f"已停止（运行 {info.duration_s:.1f} 秒）")

    # -------------------------------------------------------------------- start

    def start_server(self, *, autostart_open_ui: Optional[bool] = None) -> None:
        self.flush_all()
        built = self.build_command()
        if built is None:
            return

        if not built.validation.ok:
            message = "\n".join(f"• {e}" for e in built.validation.errors)
            self._report_validation(built)
            messagebox.showerror("无法启动", "参数校验失败：\n\n" + message, parent=self.root)
            return
        if built.validation.warnings:
            self._report_validation(built)

        # Warn about an occupied port and offer the next free one.
        host = str(self.config.get("--host", "127.0.0.1"))
        port = self.effective_port()
        if not cmdline.port_available(host, port):
            suggestion = cmdline.suggest_port(host, port + 1)
            if messagebox.askyesno(
                "端口被占用",
                f"{host}:{port} 已被占用或无法绑定。\n\n是否改用端口 {suggestion}？",
                parent=self.root,
            ):
                self.set_value("--port", suggestion)
                self.flush_all()
                built = self.build_command()
                if built is None or not built.validation.ok:
                    return
            else:
                return

        for flag in built.unknown:
            self.console_tab.append(f"[gui] 忽略本 build 不支持的参数：{flag}")

        log_path = self.store.log_path("launch")
        exe = self.layout.binary("llama-server.exe")
        self.parser.reset()
        self._listen_addr = None
        self.log_history.clear()
        self.console_tab.clear()
        self.console_tab.append(f"【启动】{time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.console_tab.append(f"【日志文件】{log_path}")

        ok, error = self.runner.start(
            exe,
            built.argv,
            cwd=self.layout.bin_dir,
            env=built.env,
            log_path=log_path,
        )
        if not ok:
            self.console_tab.append(f"【失败】{error}")
            messagebox.showerror("启动失败", error, parent=self.root)
            return

        # Remember this model for the "recent" list.
        model = self.config.get("--model")
        if model:
            self.config.push_recent(str(model))
            self.save_config(quiet=True)

        self._stop_requested = False
        self.runner.set_state(State.LOADING)
        self._start_health_watch(built, autostart_open_ui)

        def confirm_alive() -> None:
            # A bad argument or a missing DLL kills the process within
            # milliseconds.  Without this check the UI would sit on "正在加载模型…"
            # until the health watch timed out, which reads like a hang.
            if self._stop_requested or self.runner.running:
                return
            info = self.runner.poll()
            if info is None:
                self.console_tab.append("[gui] 进程已退出，未进入就绪状态")

        self.root.after(400, confirm_alive)

    def _start_health_watch(self, built: cmdline.BuiltCommand, autostart_open_ui: Optional[bool]) -> None:
        client = self.make_client()
        timeout_s = float(self.config.ui.get("load_timeout_s", 900))
        want_ui = autostart_open_ui
        if want_ui is None:
            want_ui = bool(self.config.ui.get("auto_open_webui", True))

        def work() -> None:
            health = client.wait_ready(
                timeout_s=timeout_s,
                interval_s=0.75,
                should_stop=lambda: self._stop_requested or not self.runner.running,
                on_tick=lambda h: self.post(lambda: self._note_health(h)),
                # Follow the address the server reports in its log.
                retarget=self.make_client,
            )
            self.post(lambda: self._on_ready(health, want_ui))

        self._waiting_thread = threading.Thread(target=work, name="health-watch", daemon=True)
        self._waiting_thread.start()

    def _note_health(self, health: httpclient.Health) -> None:
        """Called (on the Tk thread) for each poll while waiting for readiness."""
        self.api_tab.update_client()

    def _on_ready(self, health: httpclient.Health, open_ui: bool) -> None:
        self._waiting_thread = None
        if self._stop_requested or not self.runner.running:
            return
        if health.ready:
            self.runner.set_state(State.READY)
            self.status("服务已就绪")
            self.console_tab.append("")
            self.console_tab.append(f"【就绪】{self.make_client().web_ui_url}")
            self.api_tab.update_client()
            self._fetch_props_async()
            if open_ui:
                self.open_web_ui()
            return
        # Not ready and not stopping: either it timed out or it died.
        if health.loading:
            self.console_tab.append(
                f"【超时】等待加载超过 {self.config.ui.get('load_timeout_s', 900)} 秒仍未就绪"
            )
            self.status("加载超时（进程仍在运行）")
        else:
            self.console_tab.append(f"【未就绪】{health.describe()} {health.detail}")

    def _fetch_props_async(self) -> None:
        client = self.make_client()

        def work() -> None:
            result = client.props()
            self.post(lambda: self._absorb_props(result))

        threading.Thread(target=work, name="props", daemon=True).start()

    def _absorb_props(self, result: httpclient.HttpResult) -> None:
        if not result.ok or not isinstance(result.data, dict):
            return
        data = result.data
        settings = data.get("default_generation_settings") or {}
        n_ctx = settings.get("n_ctx")
        if isinstance(n_ctx, int):
            self.console_tab.append(f"[gui] 服务实际上下文 n_ctx = {n_ctx}")
        build = data.get("build_info")
        if build:
            self.build_info = str(build)

    # --------------------------------------------------------------------- stop

    def stop_server(self) -> None:
        """Request a stop and return immediately.

        The escalation chain in :meth:`ProcessRunner.stop` can legitimately take
        ten seconds or more (polite wait, Ctrl+Break, terminate, taskkill), so it
        must never run on the Tk main thread: doing so froze the window and left
        it unresponsive until the child was gone.
        """
        if not self._begin_stop():
            self.status("服务未在运行")

    def _begin_stop(self, on_done: Optional[Callable[[], None]] = None) -> bool:
        """Start an asynchronous shutdown, calling *on_done* when it finishes.

        Returns True when a stop is in flight (either newly started or already
        running).  Multiple callbacks may be registered; they all fire once the
        process is gone.
        """
        if on_done is not None:
            self._stop_callbacks.append(on_done)

        thread = self._stop_thread
        if thread is not None and thread.is_alive():
            return True

        if not self.runner.running:
            # Nothing to stop: the callbacks are still honoured so a restart
            # request cannot get stuck waiting for an event that will not come.
            self._drain_stop_callbacks()
            return False

        self._stop_requested = True
        self.status("正在停止…")
        # Reflect it immediately rather than waiting for the worker's report, so
        # the buttons disable on the same frame as the click.
        self.runner.set_state(State.STOPPING)
        self._apply_state(State.STOPPING, None)

        def work() -> None:
            info = self.runner.stop()
            self.post(lambda: self._on_stop_finished(info))

        self._stop_thread = threading.Thread(target=work, name="server-stop", daemon=True)
        self._stop_thread.start()
        return True

    def _on_stop_finished(self, info: Optional[ExitInfo]) -> None:
        self._stop_thread = None
        if info is not None and not info.crashed:
            self.status(f"已停止（运行 {info.duration_s:.1f} 秒）")
        self._drain_stop_callbacks()

    def _drain_stop_callbacks(self) -> None:
        callbacks, self._stop_callbacks = self._stop_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except tk.TclError:
                pass
            except Exception as exc:  # noqa: BLE001
                self.console_tab.append(f"[gui] 停止完成回调异常：{exc!r}")

    def restart_server(self) -> None:
        """Stop the current server (if any), then start again.

        Non-blocking: the stop happens on a worker thread and the start is
        triggered by its completion callback, so the window stays responsive for
        the whole restart.
        """
        if not self.runner.running:
            self.start_server()
            return

        self.flush_all()
        self.status("正在重启…")
        # _stop_requested stays True so the outgoing health watch cannot mark the
        # server ready after we asked it to stop; start_server clears it.
        self._begin_stop(on_done=self._wait_port_then_start)

    def _wait_port_then_start(self, attempt: int = 0) -> None:
        """Start once the listener port is actually free again.

        The old process has exited by this point, but the port can take a
        moment to become bindable.  Waiting avoids showing the "port in use"
        prompt during a normal restart.
        """
        host = str(self.config.get("--host", "127.0.0.1"))
        port = self.effective_port()
        if not cmdline.port_available(host, port) and attempt < 15:
            self.root.after(200, lambda: self._wait_port_then_start(attempt + 1))
            return
        if attempt >= 15:
            self.console_tab.append("[gui] 重启：端口仍被占用，仍尝试启动")
        self.start_server()

    def on_close(self) -> None:
        """Persist state and make sure no child process is orphaned.

        The stop runs on a worker thread and the window is destroyed from its
        completion callback, so quitting does not freeze the UI either.  A
        safety-net timer guarantees the window closes even if a thread wedges.
        """
        if self._close_finalized:
            return
        if self._closing:
            self.status("正在停止服务，请稍候…")
            return
        if self.bench_tab.running() and not messagebox.askyesno(
            "退出",
            "基准测试仍在运行。退出会结束它，是否继续？",
            parent=self.root,
        ):
            return
        if not self.runner.running:
            self._finalize_close()
            return
        if not messagebox.askyesno(
            "退出",
            "服务仍在运行。退出前会先停止它，是否继续？",
            parent=self.root,
        ):
            return

        self._closing = True
        self.status("正在停止服务…")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.state_label.configure(text="正在停止…",
                                   foreground=self.state_colors["stopping"])
        self._begin_stop(on_done=self._finalize_close)
        self.root.after(_CLOSE_LIMIT_MS, self._finalize_close)

    def _finalize_close(self) -> None:
        if self._close_finalized:
            return
        self._close_finalized = True
        # Last-ditch bound on the child: if the safety net fired, the stop thread
        # may still be running, so make one short attempt of our own rather than
        # risk orphaning the process.
        if self.runner.running:
            try:
                self.runner.stop(soft_timeout=0.5, hard_timeout=0.5)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.bench_tab.kill()
        except Exception:  # noqa: BLE001
            pass
        for tab in self.tabs:
            try:
                tab.flush()
            except Exception:  # noqa: BLE001
                pass
        self._collect_config()
        try:
            self.config.window["geometry"] = self.root.geometry()
        except tk.TclError:
            pass
        self.save_config(quiet=True)
        self.shutdown()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------- config plumbing

    def snapshot(self) -> Dict[str, Any]:
        """Current values plus any debounced-but-unapplied edits."""
        values = dict(self.config.values)
        values.update(self._pending)
        return values

    def get_value(self, flag: str) -> Any:
        if flag in self._pending:
            return self._pending[flag]
        return self.config.values.get(flag)

    def set_value(self, flag: str, value: Any, *, silent: bool = False) -> None:
        self.config.values[flag] = value
        self._pending.pop(flag, None)
        if not silent:
            self.update_preview()

    def on_field_change(self, spec: ArgSpec, value: Any) -> None:
        """Called by a field after its debounce: record and refresh."""
        self._pending[spec.flag] = value
        self.update_preview()

    def flush_all(self) -> None:
        for tab in self.tabs:
            try:
                tab.flush()
            except Exception:  # noqa: BLE001
                pass
        self._collect_config()

    def _collect_config(self) -> None:
        for flag, value in self._pending.items():
            self.config.values[flag] = value
        self._pending.clear()

    def show_help(self, spec: ArgSpec) -> None:
        self.status(f"{spec.flag} — {spec.help or spec.label}")

    def status(self, text: str) -> None:
        self.status_var.set(text)
        if not self._running:
            return
        self._cancel_after(self._status_after, "_status_after")
        self._status_after = self.root.after(_STATUS_CLEAR_MS, lambda: self.status_var.set("就绪"))

    # ---------------------------------------------------------------- command

    def build_command(self) -> Optional[cmdline.BuiltCommand]:
        self._collect_config()
        built = cmdline.build(
            self.config,
            help_doc=self.help_doc,
            base_dir=self.layout.bin_dir,
        )
        return built

    def update_preview(self) -> None:
        built = self.build_command()
        if built is None:
            return
        exe = self.layout.binary("llama-server.exe")
        text = cmdline.format_command([exe.name, *built.argv])
        if len(text) > 400:
            text = text[:397] + "…"
        self.command_var.set(text)
        self.update_estimate_soon()

    def _report_validation(self, built: cmdline.BuiltCommand) -> None:
        self.console_tab.append("")
        self.console_tab.append("【参数校验】")
        for error in built.validation.errors:
            self.console_tab.append(f"  ✕ {error}")
        for warning in built.validation.warnings:
            self.console_tab.append(f"  ⚠ {warning}")
        for info in built.validation.infos:
            self.console_tab.append(f"  · {info}")

    def copy_command(self) -> None:
        built = self.build_command()
        if built is None:
            return
        exe = str(self.layout.binary("llama-server.exe"))
        text = built.multiline(exe)
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status("命令已复制到剪贴板")
        except tk.TclError:
            pass

    def show_command_dialog(self) -> None:
        built = self.build_command()
        if built is None:
            return
        exe = str(self.layout.binary("llama-server.exe"))
        win = tk.Toplevel(self.root)
        win.title("命令详情")
        win.geometry("900x560")
        win.transient(self.root)

        box = ttk.Frame(win, padding=8)
        box.pack(fill="both", expand=True)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)
        box.rowconfigure(3, weight=1)

        ttk.Label(box, text="命令（argparse 前的形式，可复制到终端）",
                  style="Hint.TLabel").grid(row=0, column=0, sticky="w")
        text = tk.Text(box, wrap="word", font=("Consolas", 9), height=10)
        text.grid(row=1, column=0, sticky="nsew")
        text.insert("1.0", built.multiline(exe))
        text.configure(state="disabled")

        ttk.Label(box, text="环境变量", style="Hint.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))
        env_text = tk.Text(box, wrap="word", font=("Consolas", 9), height=6)
        env_text.grid(row=3, column=0, sticky="nsew")
        env_text.insert("1.0", "\n".join(
            f"{k}={v}" for k, v in sorted(built.env.items())
        ) or "（无）")
        env_text.configure(state="disabled")

        notes: List[str] = []
        if built.unknown:
            notes.append("本 build 不支持、已从命令中移除的参数：" + "、".join(built.unknown))
        if built.router_mode:
            notes.append("路由模式：未指定模型，llama-server 将按需加载。")
        if notes:
            ttk.Label(box, text="\n".join(notes), foreground="#b00020",
                      justify="left").grid(row=4, column=0, sticky="w", pady=6)

        ttk.Button(box, text="关闭", command=win.destroy).grid(row=5, column=0, sticky="e")

    # ------------------------------------------------------------------- health

    def build_default(self, flag: str) -> Any:
        """The default *this* llama.cpp build documents for *flag*, if any."""
        doc = self.help_doc
        opt = doc.get(flag) if doc is not None else None
        return opt.default_parsed() if opt is not None else None

    def effective_port(self) -> int:
        """The port the server will listen on.

        An unset ``--port`` means "llama.cpp's default", which is not a
        constant: it was 8080 for a long time and later became 9931.  So it is
        read from the installed build's ``--help`` rather than hard-coded.
        """
        explicit = _int_or(self.config.get("--port"), 0)
        if explicit:
            return explicit
        default = self.build_default("--port")
        if isinstance(default, int) and 0 < default < 65536:
            return default
        return 8080

    def make_client(self) -> httpclient.ServerClient:
        host = str(self.config.get("--host", "127.0.0.1") or "127.0.0.1")
        port = self.effective_port()
        keys = self.config.get("--api-key")
        api_key = _first_csv(keys)
        encrypted = bool(self.config.get("--ssl-cert-file") or self.config.get("--ssl-key-file"))
        # Once the server has said where it is listening, that is the truth:
        # it wins over anything derived from the config or the help text.
        listen = self._listen_addr
        if listen is not None and self.runner.running:
            host, port, encrypted = listen
        return httpclient.ServerClient(
            host, port,
            api_key=api_key,
            api_prefix=str(self.config.get("--api-prefix") or ""),
            use_ssl=encrypted,
        )

    def open_web_ui(self) -> None:
        url = self.make_client().web_ui_url
        try:
            webbrowser.open(url)
            self.status(f"已在浏览器中打开 {url}")
        except webbrowser.Error as exc:
            messagebox.showerror("打开浏览器", f"无法打开浏览器：{exc}\n\n地址：{url}", parent=self.root)

    # ------------------------------------------------------------------ helpers

    def device_lister(self) -> Callable[[], Tuple[List[str], str]]:
        from .proc import list_devices

        bench = self.layout.binary("llama-bench.exe")
        exe = bench if bench.is_file() else self.layout.binary("llama-server.exe")

        def work() -> Tuple[List[str], str]:
            devices, error = list_devices(exe)
            self._devices = devices
            if devices:
                self.store.write_cache_json("devices.json", {"devices": devices})
            return devices, error

        return work

    def enumerate_devices(self) -> None:
        if self.notebook.index("current") != self.tabs.index(self.hardware_tab):
            self.notebook.select(self.hardware_tab)
        self.hardware_tab.refresh_devices()

    def set_gguf_info(self, info: Optional[gguf.GGUFInfo]) -> None:
        self._gguf_info = info
        self.update_estimate_soon()

    def gguf_info(self) -> Optional[gguf.GGUFInfo]:
        return self._gguf_info

    def gguf_geometry_default_ctx(self) -> int:
        if self._gguf_info is None or self._gguf_info.error:
            return 8192
        return self._gguf_info.arch_get_int("context_length", 8192) or 8192

    def device_free_bytes(self) -> int:
        return int(self.config.ui.get("device_free_bytes", 0) or 0)

    def device_total_bytes(self) -> int:
        return int(self.config.ui.get("device_total_bytes", 0) or 0)

    def recent_models(self) -> List[str]:
        return list(self.config.recent_models)

    def server_running(self) -> bool:
        return self.runner.running

    def server_pid(self) -> Optional[int]:
        return self.runner.pid

    def bench_executable(self) -> Path:
        return self.layout.binary("llama-bench.exe")

    def open_log_dir(self) -> None:
        try:
            os.startfile(str(self.layout.logs_dir))  # type: ignore[attr-defined]
        except (AttributeError, OSError) as exc:
            self.status(f"无法打开目录：{exc}")

    def open_config_dir(self) -> None:
        try:
            os.startfile(str(self.layout.config_dir))  # type: ignore[attr-defined]
        except (AttributeError, OSError) as exc:
            self.status(f"无法打开目录：{exc}")

    def update_estimate_soon(self) -> None:
        """Refresh the estimate panel, coalescing bursts of edits.

        Returns immediately once the app has been shut down: scheduling an
        ``after`` callback after the window is destroyed makes Tcl report
        "invalid command name" for it.
        """
        if not self._running:
            return
        self._cancel_after(self._estimate_after, "_estimate_after")
        self._estimate_after = self.root.after(250, self._do_update_estimate)

    def _do_update_estimate(self) -> None:
        self._estimate_after = None
        if not self._running:
            return
        try:
            self.hardware_tab.update_estimate()
        except tk.TclError:
            pass

    def _on_tab_changed(self, _event: tk.Event) -> None:
        try:
            current = self.notebook.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            return
        resettable = bool(getattr(current, "fields", None)) or current is self.bench_tab
        try:
            self.reset_tab_button.configure(state="normal" if resettable else "disabled")
        except tk.TclError:
            pass
        if current is self.hardware_tab:
            self.update_estimate_soon()
        elif current is self.api_tab:
            self.api_tab.update_client()

    # ------------------------------------------------------------------ presets

    def save_config(self, *, quiet: bool = False) -> None:
        self._collect_config()
        try:
            self.store.save(self.config)
            if not quiet:
                self.status("配置已保存")
        except OSError as exc:
            if not quiet:
                messagebox.showerror("保存失败", str(exc), parent=self.root)

    def save_preset(self) -> None:
        from tkinter import simpledialog

        name = simpledialog.askstring("另存为预设", "预设名称：", parent=self.root)
        if not name:
            return
        self._collect_config()
        preset = config_mod.Preset(
            name=name,
            values={k: v for k, v in self.config.values.items() if v is not None},
            description="",
            model=str(self.config.get("--model") or ""),
        )
        try:
            path = self.store.save_preset(preset)
        except OSError as exc:
            messagebox.showerror("保存预设失败", str(exc), parent=self.root)
            return
        self.status(f"预设已保存：{path.name}")

    def load_preset_dialog(self) -> None:
        presets = self.store.list_presets()
        if not presets:
            messagebox.showinfo("载入预设", "还没有任何预设。用「文件 → 另存为预设」创建一个。",
                                parent=self.root)
            return
        win = tk.Toplevel(self.root)
        win.title("载入预设")
        win.geometry("620x380")
        win.transient(self.root)
        frame = ttk.Frame(win, padding=8)
        frame.pack(fill="both", expand=True)
        listbox = tk.Listbox(frame, font=("Segoe UI", 9))
        listbox.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        bar.pack(side="right", fill="y")
        listbox.configure(yscrollcommand=bar.set)
        for preset in presets:
            detail = preset.model or preset.description or ""
            listbox.insert("end", f"{preset.name}    {detail}")

        def apply() -> None:
            selection = listbox.curselection()
            if not selection:
                return
            self.apply_preset(presets[selection[0]])
            win.destroy()

        listbox.bind("<Double-Button-1>", lambda _e: apply())
        ttk.Button(frame, text="载入", command=apply).pack(pady=4)

    def apply_preset(self, preset: config_mod.Preset) -> None:
        self.config.values = {k: v for k, v in preset.values.items()}
        self._pending.clear()
        for tab in self.tabs:
            tab.refresh()
        self.update_preview()
        self.status(f"已载入预设「{preset.name}」")

    def import_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="导入预设", filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            preset = self.store.import_preset(Path(path))
        except (OSError, ValueError) as exc:
            messagebox.showerror("导入失败", str(exc), parent=self.root)
            return
        self.apply_preset(preset)
        self.status(f"已导入并在界面中应用：{preset.name}（未写入预设目录）")

    def export_preset(self) -> None:
        self._collect_config()
        path = filedialog.asksaveasfilename(
            title="导出当前配置为预设", defaultextension=".json",
            filetypes=[("JSON 文件", "*.json")],
        )
        if not path:
            return
        preset = config_mod.Preset(
            name=Path(path).stem,
            values={k: v for k, v in self.config.values.items() if v is not None},
            model=str(self.config.get("--model") or ""),
        )
        try:
            self.store.export_preset(preset, Path(path))
            self.status(f"已导出到 {path}")
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)

    def current_tab(self) -> Optional[BaseTab]:
        try:
            widget = self.notebook.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            return None
        return widget if isinstance(widget, BaseTab) else None

    def reset_current_tab(self, *, confirm: bool = True) -> List[str]:
        """Return every parameter on the visible tab to llama.cpp's default.

        "Default" means *not set*: the keys are removed from the config, so they
        drop out of the command line and llama.cpp (and ``--fit``) decide again.
        On the model tab the model file and HF repository stay selected - they
        have no default, and clearing them would only make the server
        unstartable.  Returns the flags that were cleared.
        """
        tab = self.current_tab()
        if tab is None:
            return []

        if tab is self.bench_tab:
            if self.bench_tab.form_is_default():
                self.status("基准测试选项已经是默认值")
                return []
            if confirm and not messagebox.askyesno(
                "恢复默认", "把基准测试页的选项恢复为默认值？（模型路径保留）",
                parent=self.root,
            ):
                return []
            self.bench_tab.reset_form()
            self.status("基准测试页已恢复默认")
            return []

        flags = tab.param_flags()
        if not flags:
            self.status(f"「{tab.title}」页没有可恢复的参数")
            return []

        tab.flush()  # apply a debounced edit first, so it is counted and cleared
        current = self.snapshot()
        explicit = [f for f in flags if current.get(f) is not None]
        if not explicit:
            self.status(f"「{tab.title}」页的参数已经都是默认值")
            return []

        if confirm:
            names = []
            for flag in explicit[:12]:
                spec = argspec.get(flag)
                names.append(f"• {spec.label if spec else flag}  {flag}")
            if len(explicit) > 12:
                names.append(f"… 以及另外 {len(explicit) - 12} 项")
            note = "\n\n模型文件与 HF 仓库会保留。" if tab is self.model_tab else ""
            if not messagebox.askyesno(
                "恢复默认",
                f"清空「{tab.title}」页已设置的 {len(explicit)} 个参数，交给 llama.cpp "
                f"使用默认值？\n\n" + "\n".join(names) + note
                + ("\n\n正在运行的服务不受影响，重启后生效。" if self.runner.running else ""),
                parent=self.root,
            ):
                return []

        for flag in flags:
            self.config.values.pop(flag, None)
            self._pending.pop(flag, None)
        tab.reset_fields()
        self.update_preview()
        self.status(f"「{tab.title}」页已恢复默认（清空 {len(explicit)} 个参数）")
        return explicit

    def reset_all(self) -> None:
        if not messagebox.askyesno("重置", "清空所有已设置的参数，恢复为 llama.cpp 默认？",
                                   parent=self.root):
            return
        self.config.values.clear()
        self._pending.clear()
        for tab in self.tabs:
            tab.refresh()
        self.model_tab.analyze_async(None)
        self.update_preview()
        self.status("已重置为默认")

    # -------------------------------------------------------------------- help

    def refresh_help(self) -> None:
        from .proc import capture_help

        server = self.layout.binary("llama-server.exe")
        self.status("正在重新读取 --help…")
        self.console_tab.append("[gui] 重新读取 llama-server --help（不加载模型）")

        layout = self.layout

        def work() -> None:
            text, error = capture_help(server)
            fingerprint = binary_fingerprint(layout)
            banner = ""
            if text and not error:
                # Re-probe the version too: an explicit refresh must not keep
                # showing the build number of the binaries it replaced.
                banner, _ = _server_version(layout, fingerprint, force=True)
            self.post(lambda: self._apply_help(text, error, banner, fingerprint))

        threading.Thread(target=work, name="help-refresh", daemon=True).start()

    def _apply_help(self, text: str, error: str, banner: str = "",
                    fingerprint: str = "") -> None:
        if error or not text:
            self.status(f"读取 --help 失败：{error}")
            return
        doc = helpparse.parse_help(text, str(self.layout.binary("llama-server.exe")))
        self.help_doc = doc
        self.store.write_cache("help_server.txt", text)
        if fingerprint:
            self.store.write_cache_json("fingerprint.json", {"fingerprint": fingerprint})
        if banner:
            self._set_build_info(banner)
        self.console_tab.append(f"[gui] 参数表已刷新：共 {len(doc.options)} 个参数")
        missing = doc.missing([s.flag for s in argspec.SPECS.values()])
        if missing:
            self.console_tab.append(
                f"[gui] 本 build 未提供：{'、'.join(missing)}"
            )
        self.update_preview()
        self.status("参数表已刷新")

    def show_about(self) -> None:
        text = (
            f"llama.cpp 启动器 {__version__}\n\n"
            f"llama.cpp 目录：{self.layout.bin_dir}\n"
            f"配置目录：{self.layout.config_dir}\n"
            f"llama.cpp 版本：{self.build_info or '未知'}\n"
            f"已解析参数：{len(self.help_doc.options) if self.help_doc else 0}\n\n"
            "关于显存估算：权重与 KV cache 为根据 GGUF 元数据的计算值，"
            "计算缓冲为经验估算。最终以 llama.cpp 启动日志中的实际分配为准，"
            "--fit 始终是权威。\n\n"
            "关于安全：--tools / --agent / --ui-mcp-proxy 属实验性功能，"
            "会让服务端读写本机文件并执行命令。本例程在这些选项与"
            "非本机监听地址同时出现时会拒绝启动。\n\n"
            "关于日志：本工具只读取模型文件头，不会在未点击启动时加载任何模型。"
        )
        messagebox.showinfo("关于", text, parent=self.root)

    # --------------------------------------------------------------------- run

    def run(self) -> None:
        self.update_preview()
        self.root.mainloop()


def _int_or(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


#: Files whose size and timestamp identify one llama.cpp build.  The ``.exe``
#: is a tiny launcher stub that can be byte-identical across builds, so the
#: implementation DLLs are included.
_FINGERPRINT_FILES = ("llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml-base.dll")


def binary_fingerprint(layout: paths.Layout) -> str:
    """Cheap identity of the installed binaries, used to invalidate caches."""
    parts: List[str] = []
    for name in _FINGERPRINT_FILES:
        try:
            st = layout.binary(name).stat()
        except OSError:
            continue
        parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def _server_version(
    layout: paths.Layout, fingerprint: str = "", *, force: bool = False
) -> Tuple[str, str]:
    """Read the server version banner, cached per build (see *fingerprint*)."""
    import json

    cache = layout.cache_dir / "version.json"
    if not force and cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            banner = str(data.get("banner") or "")
            if banner and data.get("fingerprint", "") == fingerprint:
                return banner, ""
        except (OSError, ValueError, AttributeError):
            pass
    from .proc import probe_version

    banner, error = probe_version(layout.binary("llama-server.exe"))
    if banner and not error:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"banner": banner, "fingerprint": fingerprint}),
                             encoding="utf-8")
        except OSError:
            pass
    return banner, error


def _first_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    text = str(value)
    return text.split(",")[0].strip()


def main() -> int:
    """Entry point used by ``llamagui.pyw`` and ``python -m llamagui``."""
    layout = paths.layout()
    if layout.missing_required:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "找不到 llama.cpp",
            "在下列位置没有找到 llama-server.exe：\n\n"
            f"{layout.bin_dir}\n\n"
            "请把 gui 目录放在 llama.cpp 安装目录内，或编辑 "
            f"{layout.gui_dir / 'install.json'} 指定 root 字段。",
        )
        root.destroy()
        return 2
    app = App(layout)
    app.run()
    return 0

"""Notebook tabs.

Most tabs are generated straight from the argument table: :class:`ParamTab`
renders every spec in its groups, grouped into labelled sections in the order
they were declared.  The specialised tabs (model, console, API, benchmark) add
behaviour on top of that base.

Each tab talks to the application through the small :class:`TabHost` protocol,
so no tab needs to know how the app wires up process control or persistence.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Iterable, List, Optional, Sequence

from .. import argspec, cmdline, estimate, gguf, httpclient
from ..argspec import ArgSpec, Group
from .widgets import Section, ScrollableFrame, ValueField, build_section_fields


class TabHost:
    """What a tab may ask of the application."""

    def get_value(self, flag: str) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def set_value(self, flag: str, value: Any, *, silent: bool = False) -> None:  # pragma: no cover
        raise NotImplementedError

    def on_field_change(self, spec: ArgSpec, value: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def show_help(self, spec: ArgSpec) -> None:  # pragma: no cover
        raise NotImplementedError

    def status(self, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class BaseTab(ttk.Frame):
    """Common plumbing for tabs."""

    def __init__(self, master: tk.Misc, app: TabHost, title: str) -> None:
        super().__init__(master, padding=0)
        self.app = app
        self.title = title
        self.fields: List[ValueField] = []

    # ------------------------------------------------------------------ helpers

    def make_scroll(self) -> ScrollableFrame:
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        return scroll

    def add_specs(
        self,
        scroll: ScrollableFrame,
        specs: Sequence[ArgSpec],
        *,
        section_title: str = "",
        on_change: Optional[Callable[[ArgSpec, Any], None]] = None,
    ) -> List[ValueField]:
        created = build_section_fields(
            scroll,
            specs,
            get_value=self.app.get_value,
            on_change=on_change or self.app.on_field_change,
            on_help=self.app.show_help,
            section_title=section_title,
        )
        self.fields.extend(created)
        return created

    def param_flags(self) -> List[str]:
        """Flags of the llama-server parameters shown on this tab."""
        return [field.spec.flag for field in self.fields]

    def reset_fields(self) -> None:
        """Show every field on this tab as "not set"."""
        for field in self.fields:
            field.reset()

    def refresh(self) -> None:
        """Re-read every field from the config (used after preset loading)."""
        for field in self.fields:
            field.set_value(self.app.get_value(field.spec.flag), silent=True)

    def flush(self) -> None:
        for field in self.fields:
            field.flush()


class ParamTab(BaseTab):
    """A tab generated entirely from one or more :class:`Group` values."""

    def __init__(
        self,
        master: tk.Misc,
        app: TabHost,
        title: str,
        groups: Sequence[Group],
    ) -> None:
        super().__init__(master, app, title)
        self.groups = list(groups)
        scroll = self.make_scroll()
        for group in self.groups:
            for section in argspec.sections_for(group):
                specs = [
                    s for s in argspec.specs_for(group)
                    if (s.section or "") == section
                ]
                if specs:
                    self.add_specs(scroll, specs, section_title=section or title)


# ------------------------------------------------------------------ model tab


class ModelTab(BaseTab):
    """Model selection plus a live GGUF summary."""

    def __init__(self, master: tk.Misc, app: TabHost) -> None:
        super().__init__(master, app, "模型")
        self._gguf_info: Optional[gguf.GGUFInfo] = None
        #: Incremented per analysis request; stale results are discarded.
        self._analysis_ticket = 0
        #: Directory of the model the mmproj logic last saw, to detect a switch.
        self._model_dir: Optional[Path] = None
        #: Debounce handle for the HuggingFace field.  Declared before the
        #: trace is attached, because assigning to the StringVar fires the
        #: callback immediately.
        self._hf_after: Optional[str] = None

        scroll = self.make_scroll()

        # --- quick source selection ---------------------------------------
        quick = Section(scroll.body, "模型来源")
        quick.grid(sticky="ew", padx=8, pady=4)
        quick.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(quick, text="本地 GGUF 文件", width=30, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=2
        )
        path_frame = ttk.Frame(quick)
        path_frame.grid(row=row, column=1, sticky="ew", pady=2)
        path_frame.columnconfigure(0, weight=1)
        self.model_var = tk.StringVar(value=self._current_model())
        ttk.Entry(path_frame, textvariable=self.model_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_frame, text="浏览…", command=self._browse).grid(row=0, column=1, padx=2)
        ttk.Button(path_frame, text="最近", command=self._show_recent).grid(row=0, column=2)
        self.model_var.trace_add("write", lambda *_a: self._on_model_text())
        row += 1

        ttk.Label(quick, text="自动检测", width=30, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=2
        )
        detect = ttk.Frame(quick)
        detect.grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(detect, text="扫描模型目录", command=self._scan_directory).grid(row=0, column=0)
        ttk.Button(detect, text="分析该文件", command=self.analyze_now).grid(row=0, column=1, padx=4)
        row += 1

        ttk.Label(quick, text="HuggingFace 仓库", width=30, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=2
        )
        self.hf_var = tk.StringVar(value=str(self.app.get_value("--hf-repo") or ""))
        ttk.Entry(quick, textvariable=self.hf_var).grid(row=row, column=1, sticky="ew", pady=2)
        self.hf_var.trace_add("write", lambda *_a: self._on_hf_text())

        # --- GGUF summary --------------------------------------------------
        info_frame = Section(scroll.body, "模型信息（读取 GGUF 文件头，不加载权重）")
        info_frame.grid(sticky="ew", padx=8, pady=4)
        info_frame.columnconfigure(1, weight=1)
        self.info_text = tk.Text(
            info_frame, height=12, wrap="none", state="disabled",
            font=("Consolas", 9), background="#fbfbfb",
        )
        self.info_text.grid(row=0, column=0, columnspan=3, sticky="ew")
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=0, column=3, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set)

        # --- the remaining curated model arguments -------------------------
        # ``--model`` and ``--hf-repo`` are rendered as the bespoke entries
        # above; everything else in this section still needs a widget, so only
        # those two are filtered out rather than the whole section.  The
        # remaining download/cache options get their own heading so the tab does
        # not show "模型来源" twice.
        custom = {"--model", "--hf-repo"}
        retitled = {"模型来源": "下载与离线"}
        for section in argspec.sections_for(Group.MODEL):
            specs = [
                s for s in argspec.specs_for(Group.MODEL)
                if (s.section or "") == section and s.flag not in custom
            ]
            if specs:
                self.add_specs(scroll, specs,
                               section_title=retitled.get(section, section))

        # Keep the bespoke entries in step with the config, then populate the
        # info panel immediately: an empty text box reads as "broken" rather
        # than "nothing selected yet".
        self.refresh()
        self.analyze_async(self.model_var.get().strip() or None)

    # ------------------------------------------------------------- plumbing

    def _current_model(self) -> str:
        value = self.app.get_value("--model")
        return str(value) if value else ""

    def refresh(self) -> None:
        super().refresh()
        model = self._current_model()
        if self.model_var.get() != model:
            self.model_var.set(model)
        hf = str(self.app.get_value("--hf-repo") or "")
        if self.hf_var.get() != hf:
            self.hf_var.set(hf)

    def _on_model_text(self) -> None:
        text = self.model_var.get().strip()
        self.app.set_value("--model", text or None, silent=True)
        self.app.on_field_change(argspec.get("--model"), text or None)  # type: ignore[arg-type]
        self.analyze_async(text or None)

    def _on_hf_text(self) -> None:
        text = self.hf_var.get().strip()
        self.app.set_value("--hf-repo", text or None, silent=True)
        if self._hf_after is not None:
            try:
                self.after_cancel(self._hf_after)
            except (tk.TclError, ValueError):
                pass
        self._hf_after = self.after(600, self._apply_hf)

    def _apply_hf(self) -> None:
        self._hf_after = None
        text = self.hf_var.get().strip()
        self.app.on_field_change(argspec.get("--hf-repo"), text or None)  # type: ignore[arg-type]

    # --------------------------------------------------------------- browsing

    def _browse(self) -> None:
        current = self.model_var.get().strip()
        initial = str(Path(current).parent) if current else None
        chosen = filedialog.askopenfilename(
            title="选择 GGUF 模型",
            initialdir=initial,
            filetypes=[("GGUF 模型", "*.gguf"), ("所有文件", "*.*")],
        )
        if chosen:
            self.model_var.set(chosen)

    def _show_recent(self) -> None:
        recent = list(self.app.recent_models())  # type: ignore[attr-defined]
        if not recent:
            messagebox.showinfo("最近使用", "还没有使用记录。", parent=self)
            return
        self._popup_list("最近使用", recent, lambda v: self.model_var.set(v))

    def _popup_list(self, title: str, items: Sequence[str], on_pick: Callable[[str], None]) -> None:
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("720x320")
        win.transient(self.winfo_toplevel())
        frame = ttk.Frame(win, padding=8)
        frame.pack(fill="both", expand=True)
        listbox = tk.Listbox(frame, font=("Consolas", 9))
        listbox.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        bar.pack(side="right", fill="y")
        listbox.configure(yscrollcommand=bar.set)
        for item in items:
            listbox.insert("end", item)

        def choose(_event: Any = None) -> None:
            selection = listbox.curselection()
            if selection:
                on_pick(items[selection[0]])
                win.destroy()

        listbox.bind("<Double-Button-1>", choose)
        ttk.Button(win, text="选择", command=choose).pack(pady=4)

    def _scan_directory(self) -> None:
        current = self.model_var.get().strip()
        initial = str(Path(current).parent) if current else str(Path.cwd())
        directory = filedialog.askdirectory(title="选择含 GGUF 的目录", initialdir=initial)
        if not directory:
            return
        found = self._find_models(Path(directory))
        if not found:
            messagebox.showinfo("扫描结果", "该目录下没有找到 .gguf 文件。", parent=self)
            return
        labels = [self._describe_model(p) for p in found]
        lookup = dict(zip(labels, [str(p) for p in found]))
        self._popup_list("扫描结果", labels, lambda label: self.model_var.set(lookup[label]))

    @staticmethod
    def _find_models(directory: Path, limit: int = 400) -> List[Path]:
        """Find models, preferring the first shard of a multi-part set."""
        results: List[Path] = []
        try:
            entries = list(directory.iterdir())
        except OSError:
            return results
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() == ".gguf":
                if gguf.is_mmproj(entry.name):
                    continue
                shard = gguf.parse_shard_name(entry.name)
                if shard is not None and shard[0] != 1:
                    continue  # only surface the first shard
                results.append(entry)
            elif entry.is_dir():
                # One level of nesting covers the documented multi-file layout.
                for nested in list(entry.iterdir())[:limit]:
                    if nested.is_file() and nested.suffix.lower() == ".gguf" \
                            and not gguf.is_mmproj(nested.name):
                        shard = gguf.parse_shard_name(nested.name)
                        if shard is not None and shard[0] != 1:
                            continue
                        results.append(nested)
            if len(results) >= limit:
                break
        return sorted(results)

    @staticmethod
    def _describe_model(path: Path) -> str:
        try:
            size = path.stat().st_size / estimate.GiB
            return f"{path.name}   ({size:.2f} GiB)   {path.parent}"
        except OSError:
            return f"{path.name}   {path.parent}"

    # -----------------------------------------------------------------– analysis

    def analyze_now(self) -> None:
        self.analyze_async(self.model_var.get().strip() or None)

    def analyze_async(self, path: Optional[str]) -> None:
        """Parse the GGUF header on a worker thread and render the summary."""
        if not path or not path.lower().endswith(".gguf"):
            self._render_info(None, path)
            return
        target = Path(path)
        if not target.is_file():
            self._render_info(None, path)
            return
        # Every request gets a ticket; only the newest one may render.  Without
        # this a slow parse of the previous model could finish after the
        # current one and leave its header (and VRAM estimate) on screen.
        self._analysis_ticket += 1
        ticket = self._analysis_ticket
        self._render_info("loading", path)  # type: ignore[arg-type]
        # Bind to the host's thread-safe poster: calling self.after() from a
        # worker raises "main thread is not in main loop" under real mainloop().
        post = self.app.post  # type: ignore[attr-defined]

        def deliver(result: Any) -> None:
            if ticket == self._analysis_ticket:
                self._render_info(result, path)

        def work() -> None:
            try:
                info = gguf.load(str(target))
            except OSError as exc:
                post(lambda: deliver({"error": str(exc)}))
                return
            post(lambda: deliver(info))

        threading.Thread(target=work, name="gguf-probe", daemon=True).start()

    def _render_info(self, info: Any, path: Optional[str]) -> None:
        if info is None or (isinstance(info, dict) and "error" in info):
            # A synchronous result supersedes any parse still in flight.
            self._analysis_ticket += 1
        if isinstance(info, str) and info == "loading":
            text = f"正在读取 GGUF 文件头…\n{path}"
            self._set_info_text(text)
            return
        if info is None:
            model = self.app.get_value("--model")
            hf = self.app.get_value("--hf-repo")
            notes = []
            if hf:
                notes.append(f"已设置 HuggingFace 仓库：{hf}（首次启动会下载）")
            if not model and not hf:
                notes.append("尚未选择模型。")
            self._set_info_text("\n".join(notes) or "尚未选择模型。")
            self._gguf_info = None
            self.app.set_gguf_info(None)  # type: ignore[attr-defined]
            # No mmproj guessing here: this branch also runs for every keystroke
            # of a half-typed path, whose parent directory means nothing yet.
            return
        if isinstance(info, dict) and "error" in info:
            self._set_info_text(f"无法读取模型信息：{info['error']}")
            self._gguf_info = None
            self.app.set_gguf_info(None)  # type: ignore[attr-defined]
            return

        self._gguf_info = info
        self.app.set_gguf_info(info)  # type: ignore[attr-defined]
        if info.error:
            self._set_info_text(f"GGUF 解析警告：{info.error}")
            return

        geo = gguf.geometry_from_info(info)
        params = info.arch_get_int("n_params") or 0
        lines: List[str] = [
            f"文件      {info.path}",
            f"大小      {estimate.human_bytes(info.file_size)}",
            f"架构      {geo.architecture or '(未知)'}    名称 {info.name}",
            f"量化      {geo.quant}",
            f"层数      {geo.n_layer}",
            f"嵌入维度  {geo.n_embd}    注意力头 {geo.n_head}    KV 头 {geo.n_head_kv}",
            f"训练上下文 {geo.n_ctx_train}",
            f"张量数量  {info.tensor_count}    权重合计 {estimate.human_bytes(info.total_tensor_bytes)}",
        ]
        if geo.uses_swa:
            lines.append(f"滑动窗口  {geo.n_swa}（KV cache 会按窗口折算）")
        if geo.is_moe:
            lines.append(f"MoE       专家数 {geo.n_expert}，每次激活 {geo.n_expert_used}")
        if geo.is_mla:
            lines.append(f"MLA       压缩 KV 维度 {geo.kv_lora_rank} + rope {geo.rope_dim}")
        lines.append(f"张量数据起始偏移 {info.data_offset}")
        if info.tensor_rows_truncated:
            lines.append("（张量清单过长，已截断显示，体积统计仍为全量）")
        self._set_info_text("\n".join(lines))

        self._maybe_autodetect_mmproj(Path(info.path).parent)

    def _set_info_text(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def _maybe_autodetect_mmproj(self, directory: Optional[Path]) -> None:
        """Keep ``--mmproj`` consistent with the selected model's directory.

        * When the model moves to a different directory, an mmproj that sat
          beside the *previous* model belongs to that model, not this one:
          loading it with an unrelated model fails, so it is cleared.
        * When no mmproj is set, a sibling projector file is filled in.

        An mmproj the user picked from some other location is left alone.
        """
        previous, self._model_dir = self._model_dir, _resolved(directory)
        current = self.app.get_value("--mmproj")
        if current and previous is not None and self._model_dir != previous:
            if _resolved(Path(str(current)).parent) == previous:
                self._set_mmproj(None)
                self.app.status(f"已清除上一个模型目录的 mmproj：{Path(str(current)).name}")
                current = None
        if current or directory is None or not directory.is_dir():
            return
        try:
            candidates = sorted(
                p for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() == ".gguf" and gguf.is_mmproj(p.name)
            )
        except OSError:
            return
        if not candidates:
            return
        self._set_mmproj(str(candidates[0]))
        self.app.status(f"已自动填入同目录的 mmproj：{candidates[0].name}")

    def _set_mmproj(self, value: Optional[str]) -> None:
        """Record *value* and show it in the mmproj field."""
        self.app.set_value("--mmproj", value, silent=True)  # type: ignore[attr-defined]
        for field in self.fields:
            if field.spec.flag == "--mmproj":
                field.set_value(value, silent=True)
        self.app.update_preview()  # type: ignore[attr-defined]


# --------------------------------------------------------------- hardware tab


class HardwareTab(BaseTab):
    """GPU offload and memory settings, with a live estimate panel."""

    def __init__(self, master: tk.Misc, app: TabHost) -> None:
        super().__init__(master, app, "显存与硬件")

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane)
        pane.add(left, weight=3)
        scroll = ScrollableFrame(left)
        scroll.pack(fill="both", expand=True)

        right = ttk.Frame(pane)
        pane.add(right, weight=2)

        for section in argspec.sections_for(Group.HARDWARE):
            specs = [
                s for s in argspec.specs_for(Group.HARDWARE) if (s.section or "") == section
            ]
            if specs:
                self.add_specs(scroll, specs, section_title=section)

        # --- device list ---------------------------------------------------
        device_frame = Section(right, "设备")
        device_frame.pack(fill="x", padx=8, pady=4)
        self.device_text = tk.Text(device_frame, height=4, wrap="word", state="disabled",
                                   font=("Consolas", 9), background="#fbfbfb")
        self.device_text.grid(row=0, column=0, sticky="ew")
        device_frame.columnconfigure(0, weight=1)
        ttk.Button(device_frame, text="刷新设备列表",
                   command=self.refresh_devices).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # --- estimate ------------------------------------------------------
        est_frame = Section(right, "显存估算（不含模型加载，纯静态推算）")
        est_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self.estimate_text = tk.Text(est_frame, height=18, wrap="word", state="disabled",
                                     font=("Consolas", 9), background="#fbfbfb")
        self.estimate_text.grid(row=0, column=0, sticky="nsew")
        est_frame.columnconfigure(0, weight=1)
        est_frame.rowconfigure(0, weight=1)

        buttons = ttk.Frame(est_frame)
        buttons.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(buttons, text="重新估算", command=self.update_estimate).pack(side="left")
        ttk.Button(buttons, text="填入建议上下文",
                   command=self._apply_suggested_context).pack(side="left", padx=4)

        self._devices: List[str] = []
        self.refresh()

    # ------------------------------------------------------------------ devices

    def refresh_devices(self) -> None:
        app = self.app
        lister = app.device_lister()  # type: ignore[attr-defined]
        post = app.post  # type: ignore[attr-defined]

        def work() -> None:
            devices, error = lister()
            post(lambda: self._render_devices(devices, error))

        threading.Thread(target=work, name="device-list", daemon=True).start()
        self._set_device_text("正在枚举设备…")

    def _render_devices(self, devices: Sequence[str], error: str) -> None:
        self._devices = list(devices)
        if error and not devices:
            self._set_device_text(f"未检测到可卸载设备。\n{error}")
            return
        self._set_device_text("\n".join(devices) if devices else "未检测到设备。")

    def _set_device_text(self, text: str) -> None:
        self.device_text.configure(state="normal")
        self.device_text.delete("1.0", "end")
        self.device_text.insert("1.0", text)
        self.device_text.configure(state="disabled")

    # ----------------------------------------------------------------- estimate

    def update_estimate(self) -> None:
        app = self.app
        info = app.gguf_info()  # type: ignore[attr-defined]
        free = app.device_free_bytes()  # type: ignore[attr-defined]
        total = app.device_total_bytes()  # type: ignore[attr-defined]

        if info is None:
            self._set_estimate_text(
                "尚未解析模型。\n\n"
                "在「模型」页选择本地 GGUF 后，这里会给出权重、KV cache、"
                "计算缓冲的静态估算，以及本机显存下可用的最大上下文。"
            )
            return
        if info.error:
            self._set_estimate_text(f"无法估算：{info.error}")
            return

        values = app.snapshot()  # type: ignore[attr-defined]
        ctx_raw = values.get("--ctx-size")
        try:
            n_ctx = int(ctx_raw) if ctx_raw not in (None, "") else 0
        except (TypeError, ValueError):
            n_ctx = 0
        parallel_raw = values.get("--parallel")
        try:
            n_parallel = int(parallel_raw) if parallel_raw not in (None, "") else 1
        except (TypeError, ValueError):
            n_parallel = 1

        actual_ctx = n_ctx or app.gguf_geometry_default_ctx()  # type: ignore[attr-defined]
        est = estimate.estimate(
            info,
            n_ctx=actual_ctx,
            ctk=str(values.get("--cache-type-k") or estimate.DEFAULT_CACHE_TYPE_K),
            ctv=str(values.get("--cache-type-v") or estimate.DEFAULT_CACHE_TYPE_V),
            n_parallel=n_parallel,
            kv_unified=values.get("--kv-unified") is not False,
            swa_full=bool(values.get("--swa-full")),
            kv_offloaded=values.get("--kv-offload") is not False,
            ubatch=_int_or(values.get("--ubatch-size"), 512),
            device_total_bytes=total,
            device_free_bytes=free,
            fit_margin_bytes=_int_or(values.get("--fit-target"), 1024) * estimate.MiB,
            gpu_layers=values.get("--gpu-layers"),
        )

        lines: List[str] = []
        for item in est.breakdown():
            marker = "~" if item.heuristic else "="
            lines.append(f"{item.label:<12}{marker} {estimate.human_bytes(item.bytes):>10}"
                         f"   {item.note}")
        lines.append("")
        lines.append(f"合计        {estimate.human_bytes(est.total_bytes)}"
                     f"   （占空闲显存 {est.total_bytes / max(1, free) * 100:.0f}%）")
        lines.append(f"空闲显存    {estimate.human_bytes(est.device_free_bytes)}"
                     f" / 总量 {estimate.human_bytes(est.device_total_bytes)}")
        lines.append(f"余量        {estimate.human_bytes(est.headroom_bytes)}")
        lines.append("")
        cap = est.geometry.n_ctx_train
        lines.append(f"本配置上下文 {actual_ctx}"
                     + (f"（模型训练上下文 {cap}）" if cap else ""))
        if est.max_ctx_fit:
            lines.append(f"建议最大上下文 {est.max_ctx_fit}"
                         f"（不预留 margin 时可到 {est.max_ctx_tight}）")
        elif free:
            lines.append("按当前设置无法为 KV cache 留出空间。")
        lines.append("")
        lines.append("模型权重与 KV 为计算值；计算缓冲为经验估算（~ 标记），")
        lines.append("最终以 llama.cpp 启动日志的实际分配为准。")
        for note in est.notes:
            lines.append(f"· {note}")
        for warning in est.warnings:
            lines.append(f"⚠ {warning}")
        if free and not est.fits:
            lines.append("")
            lines.append("⚠ 预计超出可用显存：可降低上下文、改用 q8_0 KV、")
            lines.append("  调小 -ub，或留空 -ngl 让 --fit 自动决定。")
        self._set_estimate_text("\n".join(lines))

    def _set_estimate_text(self, text: str) -> None:
        self.estimate_text.configure(state="normal")
        self.estimate_text.delete("1.0", "end")
        self.estimate_text.insert("1.0", text)
        self.estimate_text.configure(state="disabled")

    def _apply_suggested_context(self) -> None:
        info = self.app.gguf_info()  # type: ignore[attr-defined]
        if info is None or info.error:
            messagebox.showinfo("显存估算", "请先选择并解析一个模型。", parent=self)
            return
        # Recompute the suggestion the same way the panel did.
        values = self.app.snapshot()  # type: ignore[attr-defined]
        suggestion = estimate.suggest_context(
            info,
            device_free_bytes=self.app.device_free_bytes(),  # type: ignore[attr-defined]
            ctk=str(values.get("--cache-type-k") or estimate.DEFAULT_CACHE_TYPE_K),
            ctv=str(values.get("--cache-type-v") or estimate.DEFAULT_CACHE_TYPE_V),
            ubatch=_int_or(values.get("--ubatch-size"), 512),
            n_parallel=_int_or(values.get("--parallel"), 1),
            gpu_layers=values.get("--gpu-layers"),
        )
        if not suggestion:
            messagebox.showwarning(
                "显存估算",
                "按当前设置算不出可用的上下文，请先降低 KV 类型精度或调整 -ngl。",
                parent=self,
            )
            return
        self.app.set_value("--ctx-size", suggestion)
        self.refresh()
        self.update_estimate()
        self.app.status(f"已填入建议上下文 {suggestion}")


def _resolved(path: Optional[Path]) -> Optional[Path]:
    """*path* normalised for comparison, or ``None``."""
    if path is None:
        return None
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _int_or(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------- console tab


class ConsoleTab(BaseTab):
    """Live process output with filtering, pausing, and export."""

    #: Colour per log level.
    TAGS = {
        "error": "#b00020",
        "warn": "#8a6d00",
        "info": "#202020",
        "debug": "#666666",
        "gui": "#005a9e",
        "facts": "#00695c",
    }

    def __init__(self, master: tk.Misc, app: TabHost, max_lines: int = 20000) -> None:
        super().__init__(master, app, "控制台")
        self.max_lines = max_lines
        self.paused = False
        self.filter_text = ""
        self._truncated = False
        self._buffer: List[Any] = []

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=(6, 0))

        self.pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="暂停", variable=self.pause_var,
                        command=self._toggle_pause).pack(side="left")

        ttk.Label(bar, text="过滤").pack(side="left", padx=(8, 2))
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(bar, textvariable=self.filter_var, width=24)
        filter_entry.pack(side="left")
        self.filter_var.trace_add("write", lambda *_a: self._on_filter())

        self.hide_debug = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="隐藏调试", variable=self.hide_debug,
                        command=self.rebuild).pack(side="left", padx=6)

        ttk.Button(bar, text="清屏", command=self.clear).pack(side="right")
        ttk.Button(bar, text="导出日志", command=self.export).pack(side="right", padx=4)
        ttk.Button(bar, text="打开日志目录", command=self._open_log_dir).pack(side="right")

        self.facts_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.facts_var, foreground="#00695c").pack(
            fill="x", padx=8, pady=(2, 0)
        )

        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.text = tk.Text(text_frame, wrap="none", font=("Consolas", 9),
                            state="disabled", background="#1e1e1e", foreground="#dddddd")
        self.text.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

        # Colour scheme that stays readable on the dark background.
        self.TAGS.update({
            "error": "#ff6b6b",
            "warn": "#ffd166",
            "info": "#dddddd",
            "debug": "#8a8a8a",
            "gui": "#4fc3f7",
            "facts": "#7ee787",
        })
        for tag, colour in self.TAGS.items():
            self.text.tag_configure(tag, foreground=colour)

        ttk.Button(bar, text="复制全部", command=self._copy_all).pack(side="right", padx=4)

    # ------------------------------------------------------------------- input

    def append(self, line: Any) -> None:
        """Accept a :class:`~llamagui.proc.LogLine` or a plain string."""
        text = getattr(line, "text", line)
        stream = getattr(line, "stream", "stdout")
        parser = getattr(self.app, "parser", None)
        level = "info"
        if stream == "gui":
            level = "gui"
        elif parser is not None:
            level = getattr(self.app, "last_level", "info")
        self._write(str(text), level)

    def append_parsed(self, event: Any) -> None:
        """Append an already-parsed :class:`~llamagui.logparse.LogEvent`."""
        self._write(event.text, "gui" if event.milestone == "listening" else event.level)

    def append_batch(self, events: Iterable[Any]) -> None:
        for event in events:
            self.append_parsed(event)

    def _write(self, text: str, level: str) -> None:
        if self.paused:
            self._buffer.append((text, level))
            return
        if self.filter_text and self.filter_text.lower() not in text.lower():
            return
        if self.hide_debug.get() and level == "debug":
            return
        self.text.configure(state="normal")
        self.text.insert("end", text + "\n", level)
        self._trim()
        self.text.configure(state="disabled")
        self.text.see("end")

    def _trim(self) -> None:
        # Keep the widget bounded: a long generation can emit hundreds of
        # thousands of lines and an unbounded Text widget eventually stalls.
        try:
            line_count = int(self.text.index("end-1c").split(".")[0])
        except (tk.TclError, ValueError):
            return
        if line_count <= self.max_lines:
            return
        excess = line_count - self.max_lines
        self.text.delete("1.0", f"{excess + 1}.0")
        self._truncated = True

    # ------------------------------------------------------------------ actions

    def _toggle_pause(self) -> None:
        self.paused = self.pause_var.get()
        if not self.paused and self._buffer:
            buffered, self._buffer = self._buffer, []
            for text, level in buffered:
                self._write(text, level)

    def _on_filter(self) -> None:
        self.filter_text = self.filter_var.get().strip()
        self.rebuild()

    def rebuild(self) -> None:
        """Re-render from the app's retained log history."""
        history = getattr(self.app, "log_history", None)
        if history is None:
            return
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._truncated = False
        for event in list(history):
            self.append_parsed(event)

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._truncated = False

    def set_facts(self, text: str) -> None:
        suffix = "  （显示已截断，完整日志见磁盘文件）" if self._truncated else ""
        self.facts_var.set(f"{text}{suffix}" if text else "")

    def export(self) -> None:
        default = "llamagui-log.txt"
        path = filedialog.asksaveasfilename(
            title="导出控制台内容", defaultextension=".txt",
            initialfile=default, filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            content = self.text.get("1.0", "end-1c")
            Path(path).write_text(content, encoding="utf-8")
        except (OSError, tk.TclError) as exc:
            messagebox.showerror("导出失败", str(exc), parent=self)
            return
        self.app.status(f"已导出到 {path}")

    def _copy_all(self) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(self.text.get("1.0", "end-1c"))
            self.app.status("控制台内容已复制到剪贴板")
        except tk.TclError:
            pass

    def _open_log_dir(self) -> None:
        opener = getattr(self.app, "open_log_dir", None)
        if opener is not None:
            opener()


# ------------------------------------------------------------------- API tab


class ApiTab(BaseTab):
    """Endpoint reference plus ready-to-copy client snippets."""

    def __init__(self, master: tk.Misc, app: TabHost) -> None:
        super().__init__(master, app, "API 参考")

        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="服务地址", width=12, anchor="w").grid(row=0, column=0, sticky="w")
        self.base_var = tk.StringVar(value="")
        entry = ttk.Entry(top, textvariable=self.base_var, state="readonly")
        entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="复制", command=lambda: self._copy(self.base_var.get())
                   ).grid(row=0, column=2, padx=2)
        ttk.Button(top, text="打开 Web UI", command=self._open_web_ui
                   ).grid(row=0, column=3, padx=2)

        self.state_var = tk.StringVar(value="服务未启动")
        ttk.Label(top, textvariable=self.state_var, foreground="#005a9e").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )

        mid = ttk.Frame(self, padding=(8, 0))
        mid.pack(fill="both", expand=True)

        end_frame = Section(mid, "接口清单")
        end_frame.pack(side="left", fill="both", expand=True)
        end_frame.columnconfigure(0, weight=1)
        end_frame.rowconfigure(0, weight=1)
        self.endpoint_text = tk.Text(end_frame, wrap="none", font=("Consolas", 9),
                                     state="disabled", height=18, background="#fbfbfb")
        self.endpoint_text.grid(row=0, column=0, sticky="nsew")
        end_scroll = ttk.Scrollbar(end_frame, orient="vertical", command=self.endpoint_text.yview)
        end_scroll.grid(row=0, column=1, sticky="ns")
        self.endpoint_text.configure(yscrollcommand=end_scroll.set)

        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        snippet = Section(right, "调用示例")
        snippet.pack(fill="both", expand=True)
        snippet.columnconfigure(0, weight=1)
        snippet.rowconfigure(0, weight=1)
        self.snippet_text = tk.Text(snippet, wrap="word", font=("Consolas", 9),
                                    state="disabled", background="#fbfbfb")
        self.snippet_text.grid(row=0, column=0, sticky="nsew")
        buttons = ttk.Frame(snippet)
        buttons.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.snippet_choice = tk.StringVar(value="curl")
        for label, value in (("curl", "curl"), ("Python openai", "python"), ("环境变量", "env")):
            ttk.Radiobutton(buttons, text=label, value=value, variable=self.snippet_choice,
                            command=self._render_snippet).pack(side="left")
        ttk.Button(buttons, text="复制", command=self._copy_snippet).pack(side="right")

        actions = ttk.Frame(right, padding=(0, 6))
        actions.pack(fill="x")
        ttk.Button(actions, text="检测连接", command=self._probe).pack(side="left")
        ttk.Button(actions, text="API 自检（1 token）", command=self._self_test).pack(side="left", padx=4)
        ttk.Button(actions, text="读取 /props", command=self._fetch_props).pack(side="left")

        self.detail_var = tk.StringVar(value="")
        detail = ttk.Label(right, textvariable=self.detail_var, wraplength=460,
                           justify="left", foreground="#444444")
        detail.pack(fill="x")

        self._render_endpoints()
        self._render_snippet()
        self.update_client()

    # ---------------------------------------------------------------- rendering

    def _render_endpoints(self) -> None:
        lines = [f"{method:<5} {path:<34} {note}" for _name, method, path, note
                 in httpclient.ENDPOINTS]
        self.endpoint_text.configure(state="normal")
        self.endpoint_text.delete("1.0", "end")
        self.endpoint_text.insert("1.0", "\n".join(lines))
        self.endpoint_text.configure(state="disabled")

    def _client(self) -> httpclient.ServerClient:
        return self.app.make_client()  # type: ignore[attr-defined]

    def update_client(self) -> None:
        client = self._client()
        self.base_var.set(client.base_url)
        running = self.app.server_running()  # type: ignore[attr-defined]
        self.state_var.set(
            f"运行中 · PID {self.app.server_pid()}" if running else "服务未启动"
        )
        self._render_snippet()

    def _render_snippet(self) -> None:
        client = self._client()
        model = str(self.app.get_value("--alias") or "").split(",")[0].strip()
        choice = self.snippet_choice.get()
        if choice == "curl":
            text = httpclient.curl_example(client, model)
        elif choice == "python":
            text = httpclient.python_example(client, model)
        else:
            text = httpclient.env_example(client, model)
        self.snippet_text.configure(state="normal")
        self.snippet_text.delete("1.0", "end")
        self.snippet_text.insert("1.0", text)
        self.snippet_text.configure(state="disabled")

    # ------------------------------------------------------------------ actions

    def _copy(self, text: str) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.app.status("已复制到剪贴板")
        except tk.TclError:
            pass

    def _copy_snippet(self) -> None:
        self._copy(self.snippet_text.get("1.0", "end-1c"))

    def _open_web_ui(self) -> None:
        self.app.open_web_ui()  # type: ignore[attr-defined]

    def _probe(self) -> None:
        client = self._client()
        post = self.app.post  # type: ignore[attr-defined]
        self.detail_var.set("正在检测…")

        def work() -> None:
            health = client.health(timeout=3.0)
            post(lambda: self.detail_var.set(
                f"健康检查：{health.describe()}"
                + (f"（HTTP {health.status}）" if health.status else "")
            ))

        threading.Thread(target=work, daemon=True).start()

    def _self_test(self) -> None:
        if not self.app.server_running():  # type: ignore[attr-defined]
            messagebox.showinfo("API 自检", "服务未启动。", parent=self)
            return
        client = self._client()
        post = self.app.post  # type: ignore[attr-defined]
        self.detail_var.set("正在发送 1 token 请求…")
        model = str(self.app.get_value("--alias") or "").split(",")[0].strip()

        def work() -> None:
            result = client.self_test(model=model)
            if result.ok:
                text = f"自检成功（{result.elapsed_ms:.0f} ms，收到 {result.data['chunks']} 个流式分片）"
            else:
                text = result.error
            post(lambda: self.detail_var.set(text))

        threading.Thread(target=work, daemon=True).start()

    def _fetch_props(self) -> None:
        if not self.app.server_running():  # type: ignore[attr-defined]
            messagebox.showinfo("服务属性", "服务未启动。", parent=self)
            return
        client = self._client()
        post = self.app.post  # type: ignore[attr-defined]

        def work() -> None:
            result = client.props()
            post(lambda: self._show_props(result))

        threading.Thread(target=work, daemon=True).start()

    def _show_props(self, result: Any) -> None:
        if not result.ok or not isinstance(result.data, dict):
            self.detail_var.set(f"读取 /props 失败：{result.message}")
            return
        data = result.data
        settings = data.get("default_generation_settings") or {}
        params = settings.get("params") or {}
        bits = [
            f"n_ctx={settings.get('n_ctx')}",
            f"slots={data.get('total_slots')}",
            f"build={data.get('build_info')}",
        ]
        if params:
            bits.append(f"temp={params.get('temperature')}")
            bits.append(f"top_k={params.get('top_k')}")
            bits.append(f"top_p={params.get('top_p')}")
        self.detail_var.set("  ".join(str(b) for b in bits))
        self.app.status("已读取 /props")


# --------------------------------------------------------------- benchmark tab


class BenchTab(BaseTab):
    """Optional ``llama-bench`` runs.  Disabled until explicitly confirmed."""

    def __init__(self, master: tk.Misc, app: TabHost) -> None:
        super().__init__(master, app, "基准测试")

        warn = ttk.Frame(self, padding=8)
        warn.pack(fill="x")
        ttk.Label(
            warn,
            text=("llama-bench 会完整加载模型并运行推理，每个测试组合耗时数十秒到数分钟，"
                  "过程中会占用大量显存与内存。"),
            wraplength=900, justify="left", foreground="#b00020",
        ).pack(anchor="w")
        ttk.Label(
            warn,
            text=("不勾选下面的确认框时，本页不会执行任何命令。日常只需要速度数据的话，"
                  "启动服务后控制台与状态栏显示的 t/s 就是真实测量值，无需基准测试。"),
            wraplength=900, justify="left", foreground="#444444",
        ).pack(anchor="w", pady=(4, 0))

        form = ttk.Frame(self, padding=(8, 0))
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self.model_var = tk.StringVar(value=str(self.app.get_value("--model") or ""))
        self.prompt_var = tk.StringVar(value="512")
        self.gen_var = tk.StringVar(value="128")
        self.repeat_var = tk.StringVar(value="3")
        self.ngl_var = tk.StringVar(value=str(self.app.get_value("--gpu-layers") or "auto"))
        self.fa_var = tk.StringVar(value=str(self.app.get_value("--flash-attn") or "auto"))
        self.ctk_var = tk.StringVar(value=str(self.app.get_value("--cache-type-k") or "f16"))
        self.ctv_var = tk.StringVar(value=str(self.app.get_value("--cache-type-v") or "f16"))
        self.output_var = tk.StringVar(value="md")

        for row, (label, var, width) in enumerate([
            ("模型路径", self.model_var, 60),
            ("prompt token 数 -p", self.prompt_var, 12),
            ("生成 token 数 -n", self.gen_var, 12),
            ("重复次数 -r", self.repeat_var, 12),
            ("GPU 层数 -ngl", self.ngl_var, 12),
            ("Flash Attention -fa", self.fa_var, 12),
            ("KV K 类型 -ctk", self.ctk_var, 12),
            ("KV V 类型 -ctv", self.ctv_var, 12),
            ("输出格式 -o", self.output_var, 12),
        ]):
            ttk.Label(form, text=label, width=22, anchor="w").grid(
                row=row, column=0, sticky="w", pady=2
            )
            widget = ttk.Combobox(form, textvariable=var, width=width) if var in (
                self.fa_var, self.output_var
            ) else ttk.Entry(form, textvariable=var, width=width)
            if var is self.fa_var:
                widget.configure(values=["auto", "on", "off"], state="readonly")
            if var is self.output_var:
                widget.configure(values=["md", "csv", "json", "jsonl"], state="readonly")
            widget.grid(row=row, column=1, sticky="w", pady=2)

        ttk.Button(form, text="选择模型文件…", command=self._pick_model).grid(
            row=9, column=0, sticky="w", pady=4
        )

        confirm = ttk.Frame(self, padding=8)
        confirm.pack(fill="x")
        self.confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            confirm, text="我确认要完整加载模型并进行基准测试",
            variable=self.confirm_var, command=self._update_buttons,
        ).pack(side="left")

        actions = ttk.Frame(self, padding=(8, 0))
        actions.pack(fill="x")
        self.preview_button = ttk.Button(actions, text="预览命令", command=self._preview)
        self.preview_button.pack(side="left")
        self.run_button = ttk.Button(actions, text="开始测试", command=self._run)
        self.run_button.pack(side="left", padx=4)
        ttk.Button(actions, text="停止", command=self._stop).pack(side="left")
        self._update_buttons()

        self.command_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.command_var, font=("Consolas", 9),
                  wraplength=1000, justify="left", foreground="#005a9e").pack(
            fill="x", padx=8, pady=4
        )

        self.output = tk.Text(self, wrap="none", font=("Consolas", 9), height=18,
                              state="disabled", background="#1e1e1e", foreground="#dddddd")
        self.output.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._proc: Optional[Any] = None
        self._reader: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ helpers

    def _pick_model(self) -> None:
        chosen = filedialog.askopenfilename(
            title="选择用于基准测试的 GGUF",
            filetypes=[("GGUF 模型", "*.gguf"), ("所有文件", "*.*")],
        )
        if chosen:
            self.model_var.set(chosen)

    def _update_buttons(self) -> None:
        state = "normal" if self.confirm_var.get() else "disabled"
        self.run_button.configure(state=state)

    #: Form defaults; "auto"/"f16"/"md" mean "let llama-bench decide".
    FORM_DEFAULTS = (
        ("prompt_var", "512"), ("gen_var", "128"), ("repeat_var", "3"),
        ("ngl_var", "auto"), ("fa_var", "auto"), ("ctk_var", "f16"),
        ("ctv_var", "f16"), ("output_var", "md"),
    )

    def form_is_default(self) -> bool:
        return all(getattr(self, name).get() == value for name, value in self.FORM_DEFAULTS)

    def reset_form(self) -> None:
        """Restore the benchmark options; the chosen model file is kept."""
        for name, value in self.FORM_DEFAULTS:
            getattr(self, name).set(value)
        self.confirm_var.set(False)
        self._update_buttons()
        self.command_var.set("")

    def build_argv(self) -> List[str]:
        argv: List[str] = ["--model", self.model_var.get().strip()]
        for flag, var, default in (
            ("-p", self.prompt_var, "512"),
            ("-n", self.gen_var, "128"),
            ("-r", self.repeat_var, "3"),
        ):
            value = var.get().strip()
            if value and value != default:
                argv.extend([flag, value])
        if self.ngl_var.get().strip() not in ("", "auto"):
            argv.extend(["-ngl", self.ngl_var.get().strip()])
        if self.fa_var.get().strip() not in ("", "auto"):
            argv.extend(["-fa", self.fa_var.get().strip()])
        if self.ctk_var.get().strip() not in ("", "f16"):
            argv.extend(["-ctk", self.ctk_var.get().strip()])
        if self.ctv_var.get().strip() not in ("", "f16"):
            argv.extend(["-ctv", self.ctv_var.get().strip()])
        out = self.output_var.get().strip()
        if out and out != "md":
            argv.extend(["-o", out])
        return argv

    # ------------------------------------------------------------------ actions

    def _preview(self) -> None:
        argv = self.build_argv()
        exe = self.app.bench_executable()  # type: ignore[attr-defined]
        self.command_var.set(cmdline.format_command([str(exe), *argv]))
        self.app.status("已生成基准测试命令（未执行）")

    def _run(self) -> None:
        if not self.confirm_var.get():
            return
        if self.running():
            messagebox.showinfo("基准测试", "已有一个基准测试在运行，请先停止。", parent=self)
            return
        model = self.model_var.get().strip()
        if not model or not Path(model).is_file():
            messagebox.showerror("基准测试", "请选择一个存在的 GGUF 模型文件。", parent=self)
            return
        argv = self.build_argv()
        exe = Path(self.app.bench_executable())  # type: ignore[attr-defined]
        if not exe.is_file():
            messagebox.showerror("基准测试", f"未找到 {exe}。", parent=self)
            return
        self.command_var.set(cmdline.format_command([str(exe), *argv]))
        self._clear_output()
        self._write_output(f"$ {cmdline.format_command([str(exe), *argv])}\n")

        import subprocess

        from ..proc import CREATIONFLAGS_NO_WINDOW

        try:
            self._proc = subprocess.Popen(
                [str(exe), *argv],
                cwd=str(exe.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATIONFLAGS_NO_WINDOW,
            )
        except OSError as exc:
            messagebox.showerror("基准测试", f"启动失败：{exc}", parent=self)
            return
        self.app.status("基准测试运行中…")
        post = self.app.post  # type: ignore[attr-defined]

        def drain() -> None:
            proc = self._proc
            if proc is None or proc.stdout is None:
                return
            for line in proc.stdout:
                post(lambda text=line.rstrip("\n"): self._write_output(text))
            code = proc.wait()
            post(lambda: self.app.status(f"基准测试结束，返回码 {code}"))

        self._reader = threading.Thread(target=drain, name="llama-bench", daemon=True)
        self._reader.start()

    def _stop(self) -> None:
        if self.running():
            self.kill()
            self.app.status("已请求停止基准测试")

    def kill(self, wait_s: float = 2.0) -> None:
        """End a running llama-bench, e.g. when the window closes.

        A benchmark holds the whole model in VRAM, so leaving it behind as an
        orphan after the GUI exits is not acceptable.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=wait_s)
        except (OSError, ValueError):
            pass
        except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
            import subprocess

            from ..proc import CREATIONFLAGS_NO_WINDOW

            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=10,
                               creationflags=CREATIONFLAGS_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                pass

    def _clear_output(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def _write_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text + "\n")
        self.output.configure(state="disabled")
        self.output.see("end")

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

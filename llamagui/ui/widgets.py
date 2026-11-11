"""Reusable tkinter building blocks.

Everything here is driven by the declarative :mod:`llamagui.argspec` table, so
adding an argument to the table makes it appear in the UI with no further work.

The important contract is :class:`ValueField`: a field holds a *pending* value
that is only flushed into the config after a short debounce.  This behaviour is
what makes the command preview and the live VRAM estimate cheap, while keeping
every keystroke off the config file.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, List, Optional, Sequence

from ..argspec import ArgSpec, Kind


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container.

    tkinter has no built-in scrollable frame, and the standard recipe
    (a ``Canvas`` with an inner ``Frame``) has three classic pitfalls that this
    class handles explicitly:

    * the inner frame must be resized to the canvas width so labels wrap;
    * the scroll region must be recomputed whenever content changes;
    * the mouse wheel must be bound only while the pointer is over the canvas,
      otherwise scrolling anywhere in the app moves this frame.
    """

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.body = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_body_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        # Keep the inner frame exactly as wide as the visible area.
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event: tk.Event) -> None:
        delta = 0
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "delta", 0):
            # Windows delivers multiples of 120; macOS delivers small values.
            delta = -1 if event.delta > 0 else 1
        if delta:
            self.canvas.yview_scroll(delta, "units")

    def clear(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()


class Section(ttk.LabelFrame):
    """A titled group of fields."""

    def __init__(self, master: tk.Misc, title: str, **kwargs: Any) -> None:
        super().__init__(master, text=title, padding=(10, 6), **kwargs)
        self.columnconfigure(1, weight=1)
        self._row = 0

    def add(self, *widgets: tk.Widget) -> None:
        for widget in widgets:
            widget.grid(row=self._row, column=0, sticky="w", pady=2)
            self._row += 1


class ValueField:
    """Edits one :class:`ArgSpec` and reports changes with a debounce.

    The field never writes to the config on every keystroke.  Instead it calls
    ``on_change(spec, pending_value)`` (default 400 ms debounce), and
    ``on_flush`` immediately before the value is needed for a launch.  Callers
    therefore get one authoritative read at exactly the right moment.
    """

    def __init__(
        self,
        parent: ttk.Frame,
        spec: ArgSpec,
        *,
        row: int,
        get_value: Callable[[str], Any],
        on_change: Callable[[ArgSpec, Any], None],
        debounce_ms: int = 400,
        label_width: int = 30,
        help_width: int = 46,
        on_help: Optional[Callable[[ArgSpec], None]] = None,
    ) -> None:
        self.spec = spec
        self.get_value = get_value
        self.on_change = on_change
        self.debounce_ms = debounce_ms
        self.label_width = label_width
        self.help_width = help_width

        self.var: tk.Variable
        self.widget: tk.Widget
        self.extra_widgets: List[tk.Widget] = []
        self._after_id: Optional[str] = None
        self._suppress = False

        self.label = ttk.Label(parent, text=spec.label, width=label_width, anchor="w")
        self.label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)

        kind = spec.kind
        initial = get_value(spec.flag)
        self._build(parent, row, kind, initial)

        hint = self._hint_text()
        self.hint = ttk.Label(parent, text=hint, width=help_width, anchor="w",
                              foreground="#666666")
        self.hint.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=2)

        if spec.help and on_help is not None:
            self.label.configure(cursor="hand2")
            self.label.bind("<Button-1>", lambda _e: on_help(spec))
            self.label.bind("<Enter>", lambda _e: self._show_help(on_help, spec))
        if spec.danger:
            self.label.configure(foreground="#b00020")
        elif spec.experimental:
            self.label.configure(foreground="#8a6d00")

    # ------------------------------------------------------------------ build

    def _build(self, parent: ttk.Frame, row: int, kind: Kind, initial: Any) -> None:
        spec = self.spec

        if kind is Kind.BOOL:
            # Tri-state is expressed without a third widget: unchecked means
            # "not set" unless a value is already present in the config.
            self.var = tk.BooleanVar(value=bool(initial) if initial is not None else False)
            self.widget = ttk.Checkbutton(
                parent, text="启用/关闭", variable=self.var, command=self._emit
            )
            self.widget.grid(row=row, column=1, sticky="w", pady=2)

        elif kind in (Kind.CHOICE, Kind.TRISTATE):
            self.var = tk.StringVar(value="" if initial is None else str(initial))
            values = [""] + list(spec.choices)
            self.widget = ttk.Combobox(
                parent, textvariable=self.var, values=values, state="readonly", width=22
            )
            self.widget.grid(row=row, column=1, sticky="w", pady=2)
            self.widget.bind("<<ComboboxSelected>>", lambda _e: self._emit())

        elif kind is Kind.REPEAT:
            self.var = tk.StringVar(value=self._join(initial))
            self._place_entry(parent, row, width=24, masked=spec.secret)
            self.var.trace_add("write", lambda *_a: self._schedule())

        elif kind is Kind.TEXT:
            self.var = tk.StringVar(value="" if initial is None else str(initial))
            self.widget = ttk.Entry(parent, textvariable=self.var, width=24)
            self.widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.var.trace_add("write", lambda *_a: self._schedule())

        elif kind is Kind.PATH:
            self.var = tk.StringVar(value="" if initial is None else str(initial))
            frame = ttk.Frame(parent)
            entry = ttk.Entry(frame, textvariable=self.var, width=24)
            entry.grid(row=0, column=0, sticky="ew")
            frame.columnconfigure(0, weight=1)
            browse = ttk.Button(frame, text="…", width=3, command=lambda: self._browse(False))
            browse.grid(row=0, column=1)
            clear = ttk.Button(frame, text="✕", width=3, command=self._clear)
            clear.grid(row=0, column=2)
            frame.grid(row=row, column=1, sticky="ew", pady=2)
            self.widget = entry
            self.extra_widgets = [browse, clear]
            self.var.trace_add("write", lambda *_a: self._schedule())

        else:  # INT / FLOAT / STR / CSV
            self.var = tk.StringVar(value="" if initial is None else self._format(initial))
            self._place_entry(parent, row, width=24, masked=spec.secret)
            self.var.trace_add("write", lambda *_a: self._schedule())

    def _place_entry(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        width: int = 24,
        masked: bool = False,
    ) -> ttk.Entry:
        """Create the entry for this field and grid it.

        When *masked* the entry is built inside its own frame together with a
        reveal toggle, so a credential is not readable in a screenshot but the
        user can still check what they pasted.  The entry is created with the
        frame as its master rather than re-parented with ``grid(in_=...)``,
        which would mix geometry masters and is fragile.
        """
        if not masked:
            entry = ttk.Entry(parent, textvariable=self.var, width=width)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.widget = entry
            return entry

        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="ew", pady=2)
        holder.columnconfigure(0, weight=1)

        entry = ttk.Entry(holder, textvariable=self.var, width=width, show="•")
        entry.grid(row=0, column=0, sticky="ew")

        revealed = tk.BooleanVar(value=False)

        def toggle() -> None:
            entry.configure(show="" if revealed.get() else "•")

        button = ttk.Checkbutton(holder, text="显示", variable=revealed, command=toggle)
        button.grid(row=0, column=1, padx=(4, 0))

        self.widget = entry
        self.extra_widgets = [button]
        return entry

    def _join(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    def _format(self, value: Any) -> str:
        if self.spec.kind is Kind.FLOAT:
            try:
                f = float(value)
                if f == int(f) and abs(f) < 1e15:
                    return str(int(f))
                return repr(f)
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def _hint_text(self) -> str:
        spec = self.spec
        bits: List[str] = []
        if spec.default is not None and spec.kind is not Kind.BOOL:
            bits.append(f"默认 {spec.default}")
        if spec.unit:
            bits.append(spec.unit)
        if spec.minimum is not None or spec.maximum is not None:
            low = "" if spec.minimum is None else str(spec.minimum)
            high = "" if spec.maximum is None else str(spec.maximum)
            bits.append(f"[{low}~{high}]")
        if spec.experimental:
            bits.append("实验性")
        if spec.env:
            bits.append(spec.env)
        return "  ".join(bits)

    # ----------------------------------------------------------------- events

    def _show_help(self, on_help: Callable[[ArgSpec], None], spec: ArgSpec) -> None:
        on_help(spec)

    def _schedule(self) -> None:
        if self._suppress:
            return
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except (tk.TclError, ValueError):
                pass
        self._after_id = self.widget.after(self.debounce_ms, self._emit)

    def _emit(self) -> None:
        self._after_id = None
        self.on_change(self.spec, self.pending())
        self._refresh_marker()

    def flush(self) -> None:
        """Apply any debounced edit immediately."""
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except (tk.TclError, ValueError):
                pass
            self._after_id = None
            self._emit()

    # --------------------------------------------------------------- read/write

    def pending(self) -> Any:
        """The value this field currently represents.

        Returns ``None`` for "not set", which keeps the argument out of the
        command line so llama.cpp defaults stay in charge.
        """
        kind = self.spec.kind
        if kind is Kind.BOOL:
            return bool(self.var.get())
        raw = str(self.var.get()).strip()
        if raw == "":
            return None
        if kind is Kind.REPEAT:
            items = [p.strip() for p in raw.split(",") if p.strip()]
            return items if items else None
        if kind is Kind.CSV:
            return raw
        if kind in (Kind.INT, Kind.FLOAT):
            # Leave unparsable input as a string; validation reports it.
            try:
                return int(raw) if kind is Kind.INT else float(raw)
            except ValueError:
                return raw
        return raw

    def set_value(self, value: Any, *, silent: bool = True) -> None:
        """Programmatically set the field, e.g. when loading a preset."""
        self._suppress = silent
        try:
            if self.spec.kind is Kind.BOOL:
                self.var.set(bool(value) if value is not None else False)
            elif self.spec.kind is Kind.REPEAT:
                self.var.set(self._join(value))
            elif value is None:
                self.var.set("")
            elif self.spec.kind in (Kind.INT, Kind.FLOAT):
                self.var.set(self._format(value))
            else:
                self.var.set(str(value))
        finally:
            self._suppress = False
        self._refresh_marker()

    def reset(self) -> None:
        """Return the field to "not set"."""
        self.set_value(None, silent=True)

    def _refresh_marker(self) -> None:
        """Bold the label when the user has actually set a value."""
        try:
            current = self.get_value(self.spec.flag)
        except Exception:  # noqa: BLE001 - marker is cosmetic only
            return
        font = ("Segoe UI", 9, "bold") if current is not None else ("Segoe UI", 9, "normal")
        try:
            self.label.configure(font=font)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ browse

    def _browse(self, directory: bool) -> None:
        from tkinter import filedialog

        current = str(self.var.get()).strip()
        initial = current or None
        try:
            if directory:
                chosen = filedialog.askdirectory(initialdir=initial)
            else:
                chosen = filedialog.askopenfilename(
                    initialdir=initial,
                    filetypes=[
                        ("GGUF 模型", "*.gguf"),
                        ("所有文件", "*.*"),
                    ],
                )
        except tk.TclError:
            return
        if chosen:
            self.var.set(chosen)
            self._emit()

    def _clear(self) -> None:
        self.var.set("")
        self._emit()


def build_section_fields(
    parent: ScrollableFrame,
    specs: Sequence[ArgSpec],
    *,
    get_value: Callable[[str], Any],
    on_change: Callable[[ArgSpec, Any], None],
    on_help: Optional[Callable[[ArgSpec], None]] = None,
    section_title: str = "",
    debounce_ms: int = 400,
) -> List[ValueField]:
    """Lay out one labelled section containing *specs*."""
    if not specs:
        return []
    if section_title:
        frame: ttk.Frame = Section(parent.body, section_title)
        frame.grid(sticky="ew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        target = frame
    else:
        frame = ttk.Frame(parent.body)
        frame.grid(sticky="ew", padx=8, pady=4)
        frame.columnconfigure(1, weight=1)
        target = frame

    fields: List[ValueField] = []
    for index, spec in enumerate(specs):
        field = ValueField(
            target,
            spec,
            row=index,
            get_value=get_value,
            on_change=on_change,
            on_help=on_help,
            debounce_ms=debounce_ms,
        )
        fields.append(field)
    return fields


def labelled_entry(
    parent: ttk.Frame, row: int, label: str, value: str = "", width: int = 24,
    label_width: int = 30,
) -> tk.StringVar:
    """A simple label + entry pair not tied to an ArgSpec."""
    ttk.Label(parent, text=label, width=label_width, anchor="w").grid(
        row=row, column=0, sticky="w", padx=(0, 6), pady=2
    )
    var = tk.StringVar(value=value)
    ttk.Entry(parent, textvariable=var, width=width).grid(
        row=row, column=1, sticky="ew", pady=2
    )
    return var

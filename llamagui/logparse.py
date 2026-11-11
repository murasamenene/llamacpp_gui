"""Real-time parsing of llama-server output.

llama.cpp logs a great deal of useful state on startup: the context size it
ended up with (after ``--fit`` had its say), the KV cache size it allocated per
layer, how many layers were offloaded, the backends and devices, and later the
prompt-eval and generation throughput.  This module extracts those numbers so
the GUI can show measured facts next to its own pre-flight estimates.

Both output shapes are supported: the default textual log and ``--log-jsonl``.
ANSI colour codes are stripped defensively even though the launcher asks for
``--log-colors off``, because a user pasting a command into a real console will
get colours.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: CSI / OSC / single-character escape sequences.
_ANSI_RE = re.compile(
    r"\x1B(?:"
    r"\[[0-?]*[ -/]*[@-~]"      # CSI
    r"|\][^\x07\x1B]*(?:\x07|\x1B\\)"  # OSC
    r"|[@-Z\\-_]"                # Fe escapes
    r")"
)

_LEVEL_RE = re.compile(r"\b(ERROR|ERR|WARN|WARNING|INFO|DEBUG|TRACE)\b")
_INFIX_RE = re.compile(r"\b(error|warn|warning|trace|debug)\b", re.IGNORECASE)

# ---------------------------------------------------------------- text fields

_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    # n_ctx = 8192 / n_ctx_per_seq / n_batch / n_ubatch
    ("n_ctx", re.compile(r"n_ctx\s*=\s*(\d+)")),
    ("n_ctx_per_seq", re.compile(r"n_ctx_per_seq\s*=\s*(\d+)")),
    ("n_batch", re.compile(r"n_batch\s*=\s*(\d+)")),
    ("n_ubatch", re.compile(r"n_ubatch\s*=\s*(\d+)")),
    ("n_threads", re.compile(r"n_threads\s*=\s*(\d+)")),
    ("n_seq_max", re.compile(r"n_seq_max\s*=\s*(\d+)")),
    ("n_ctx_train", re.compile(r"n_ctx_train\s*=\s*(\d+)")),
    # KV cache: "KV self size = 1024.00 MiB, K (f16): 512.00, V (f16): 512.00"
    ("kv_size_mib", re.compile(r"KV self size\s*=\s*([\d.]+)\s*MiB")),
    ("kv_k_mib", re.compile(r"K\s*\(([a-z0-9_]+)\):\s*([\d.]+)")),
    ("kv_v_mib", re.compile(r"V\s*\(([a-z0-9_]+)\):\s*([\d.]+)")),
    # CUDA/Host buffer sizes: "CUDA0 model buffer size = 1234.56 MiB"
    ("buffer", re.compile(r"^(\w+) (model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB")),
    # Offload: "offloaded 33/33 layers to GPU"
    ("offloaded", re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers")),
    ("offloaded_kv", re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s*KV")),
    # Throughput
    ("prompt_per_second", re.compile(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second")),
    ("eval_per_second", re.compile(r"^\s*eval time\s*=.*?([\d.]+)\s*tokens per second")),
    ("prompt_eval_ms", re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms")),
    ("eval_ms", re.compile(r"^\s*eval time\s*=\s*([\d.]+)\s*ms")),
    # Model metadata summary lines
    ("total_params", re.compile(r"model params\s*=\s*([\d.]+)\s*([BMK])")),
    ("model_size", re.compile(r"model size\s*=\s*([\d.]+)\s*MiB")),
    ("file_type", re.compile(r"ftype\s*=\s*(\S+)")),
    ("n_layer", re.compile(r"n_layer\s*=\s*(\d+)")),
    ("n_embd", re.compile(r"n_embd\s*=\s*(\d+)")),
    # Backends / devices: "load_backend: loaded CUDA backend from ..."
    ("backend", re.compile(r"loaded\s+(\w+)\s+backend\s+from\s+(.+?)\s*$")),
    # ``CUDA0: NVIDIA GeForce RTX 3080 (20479 MiB, 19297 MiB free)``.  Requires
    # the backend prefix and a colon so the summary line ``Device 0: ...`` that
    # appears in the init banner is not mistaken for a device listing.
    ("device", re.compile(
        r"^\s*((?:CUDA|ROCm|Vulkan|SYCL|Metal|RPC)\d*|CPU)\s*:\s*(\S.*?)\s*$"
    )),
    ("startup_ms", re.compile(r"llama_server: server is listening|main: server is listening")),
)

#: Messages worth surfacing as a highlighted status line.
_MILESTONES: Tuple[Tuple[str, re.Pattern], ...] = (
    # Old builds: "server is listening on http://...", newer ones print
    # "server_on_li: listening on http://127.0.0.1:9931".
    ("listening", re.compile(r"\blistening on\s+(?:https?://)?\S+:\d+", re.IGNORECASE)),
    ("loading", re.compile(r"^\s*load_tensors:|loading model", re.IGNORECASE)),
    ("loaded", re.compile(r"model loaded|llama_model_load: model loaded", re.IGNORECASE)),
    ("warmup", re.compile(r"warming up", re.IGNORECASE)),
    ("failed", re.compile(r"failed to|error loading|out of memory|CUDA error|"
                          r"ggml_backend_cuda|alloc failed", re.IGNORECASE)),
    ("download", re.compile(r"downloading|download progress|fetching", re.IGNORECASE)),
)


#: The address the server actually bound, from its "listening on" line.
_LISTEN_URL_RE = re.compile(r"listening on\s+((?:https?://)?\S+:\d+)", re.IGNORECASE)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    if "\x1b" not in text:
        return text
    return _ANSI_RE.sub("", text)


def guess_level(line: str) -> str:
    """Classify a log line as one of ``error``/``warn``/``info``/``debug``."""
    m = _LEVEL_RE.search(line)
    if m:
        token = m.group(1).upper()
        if token in ("ERROR", "ERR"):
            return "error"
        if token in ("WARN", "WARNING"):
            return "warn"
        if token in ("DEBUG", "TRACE"):
            return "debug"
        return "info"
    lowered = line.lower()
    if "error" in lowered or "failed" in lowered or "out of memory" in lowered:
        return "error"
    if "warn" in lowered:
        return "warn"
    if "debug" in lowered or "trace" in lowered:
        return "debug"
    return "info"


@dataclass
class LogEvent:
    """One parsed log line."""

    raw: str
    text: str
    level: str = "info"
    timestamp: float = field(default_factory=time.time)
    fields: Dict[str, object] = field(default_factory=dict)
    milestone: Optional[str] = None
    json_payload: Optional[dict] = None

    @property
    def is_error(self) -> bool:
        return self.level == "error"


@dataclass
class ServerFacts:
    """Accumulated facts about the running server, used by the status bar."""

    n_ctx: Optional[int] = None
    n_ctx_per_seq: Optional[int] = None
    n_ctx_train: Optional[int] = None
    n_batch: Optional[int] = None
    n_ubatch: Optional[int] = None
    n_seq_max: Optional[int] = None
    n_layer: Optional[int] = None
    n_embd: Optional[int] = None
    kv_size_mib: Optional[float] = None
    kv_k_type: Optional[str] = None
    kv_v_type: Optional[str] = None
    kv_k_mib: Optional[float] = None
    kv_v_mib: Optional[float] = None
    offloaded_layers: Optional[int] = None
    total_layers: Optional[int] = None
    model_size_mib: Optional[float] = None
    total_params: Optional[str] = None
    file_type: Optional[str] = None
    prompt_per_second: Optional[float] = None
    eval_per_second: Optional[float] = None
    backends: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    #: Sum of every ``* buffer size`` line seen, in MiB.
    buffers_mib: Dict[str, float] = field(default_factory=dict)
    milestones: List[Tuple[str, float]] = field(default_factory=list)
    #: ``http://host:port`` the server reported it is listening on.
    listen_url: Optional[str] = None

    def merge_buffers(self, key: str, mib: float) -> None:
        self.buffers_mib[key] = self.buffers_mib.get(key, 0.0) + mib

    @property
    def buffers_total_mib(self) -> float:
        return sum(self.buffers_mib.values())

    def summary(self) -> str:
        """Short one-line description for the status bar."""
        bits: List[str] = []
        if self.n_ctx:
            bits.append(f"ctx {self.n_ctx}")
        if self.offloaded_layers is not None and self.total_layers:
            bits.append(f"GPU {self.offloaded_layers}/{self.total_layers} 层")
        if self.kv_size_mib:
            bits.append(f"KV {self.kv_size_mib:.0f} MiB")
        if self.eval_per_second:
            bits.append(f"生成 {self.eval_per_second:.1f} t/s")
        if self.prompt_per_second:
            bits.append(f"预填充 {self.prompt_per_second:.1f} t/s")
        return " · ".join(bits)


class LogParser:
    """Stateful parser: feed lines in, collect :class:`LogEvent` and facts."""

    def __init__(self) -> None:
        self.facts = ServerFacts()
        self.line_count = 0
        self.error_count = 0
        self.warning_count = 0

    # ------------------------------------------------------------------ feed

    def parse_line(self, line: str) -> Optional[LogEvent]:
        text = strip_ansi(line).rstrip("\r\n")
        if not text.strip():
            return None
        self.line_count += 1

        payload: Optional[dict] = None
        fields: Dict[str, object] = {}
        milestone: Optional[str] = None
        level = "info"

        json_event = self._try_jsonl(text)
        if json_event is not None:
            payload = json_event
            message = str(payload.get("msg") or payload.get("message") or text)
            level = self._json_level(payload)
            text = self._format_jsonl(payload, message)
            self._absorb_json(payload)
        else:
            level = guess_level(text)
            fields = self._extract(text)
            self._absorb_fields(fields)

        for name, pattern in _MILESTONES:
            if pattern.search(text):
                milestone = name
                self.facts.milestones.append((name, time.time()))
                break
        if milestone == "listening":
            url = _LISTEN_URL_RE.search(text)
            if url:
                found = url.group(1).rstrip("/.,;")
                if "://" not in found:
                    found = "http://" + found  # older builds omit the scheme
                self.facts.listen_url = found

        if level == "error":
            self.error_count += 1
        elif level == "warn":
            self.warning_count += 1

        return LogEvent(
            raw=line,
            text=text,
            level=level,
            fields=fields,
            milestone=milestone,
            json_payload=payload,
        )

    def reset(self) -> None:
        self.facts = ServerFacts()
        self.line_count = 0
        self.error_count = 0
        self.warning_count = 0

    # ---------------------------------------------------------------- helpers

    def _try_jsonl(self, text: str) -> Optional[dict]:
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _json_level(payload: dict) -> str:
        raw = payload.get("level")
        if isinstance(raw, str):
            token = raw.lower()
            if token in ("error", "err", "fatal", "critical"):
                return "error"
            if token in ("warn", "warning"):
                return "warn"
            if token in ("debug", "trace"):
                return "debug"
        # Some builds use a numeric severity.
        if isinstance(raw, int):
            if raw <= 1:
                return "error"
            if raw == 2:
                return "warn"
            if raw >= 4:
                return "debug"
        return "info"

    @staticmethod
    def _format_jsonl(payload: dict, message: str) -> str:
        stamp = payload.get("time") or payload.get("timestamp")
        stage = payload.get("func") or payload.get("source") or payload.get("scope")
        parts = []
        if isinstance(stamp, str) and stamp:
            parts.append(stamp.split("T")[-1].rstrip("Z") if "T" in stamp else stamp)
        if isinstance(stage, str) and stage:
            parts.append(stage)
        parts.append(message)
        return " ".join(parts)

    def _absorb_json(self, payload: dict) -> None:
        """Pick structured values out of a JSONL record."""
        for key in ("n_ctx", "n_ctx_per_seq", "n_ctx_train", "n_batch", "n_ubatch",
                    "n_seq_max", "n_layer", "n_embd"):
            value = payload.get(key)
            if isinstance(value, int):
                setattr(self.facts, key, value)
        if isinstance(payload.get("kv_size_mib"), (int, float)):
            self.facts.kv_size_mib = float(payload["kv_size_mib"])

    def _extract(self, text: str) -> Dict[str, object]:
        """Run every textual pattern over *text* and return named captures."""
        found: Dict[str, object] = {}
        for name, pattern in _PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            groups = m.groups()
            if name in ("kv_k_mib", "kv_v_mib"):
                # (dtype, mib) pairs
                found[name] = float(groups[1])
                found["kv_k_type" if name == "kv_k_mib" else "kv_v_type"] = groups[0]
            elif name == "buffer":
                found.setdefault("buffers", []).append(
                    (f"{groups[0]} {groups[1]}", float(groups[2]))
                )
            elif name == "offloaded":
                found["offloaded_layers"] = int(groups[0])
                found["total_layers"] = int(groups[1])
            elif name in ("backend",):
                found["backend"] = (groups[0], groups[1].strip())
            elif name == "device":
                found["device"] = (groups[0], groups[1])
            elif name == "total_params":
                found["total_params"] = f"{groups[0]}{groups[1]}"
            elif name == "startup_ms":
                found["listening"] = True
            elif len(groups) == 1:
                value: object = groups[0]
                if re.fullmatch(r"\d+", str(value)):
                    value = int(str(value))
                elif re.fullmatch(r"[\d.]+", str(value)):
                    value = float(str(value))
                found[name] = value
        return found

    def _absorb_fields(self, fields: Dict[str, object]) -> None:
        f = self.facts
        for key in ("n_ctx", "n_ctx_per_seq", "n_ctx_train", "n_batch", "n_ubatch",
                    "n_seq_max", "n_layer", "n_embd", "offloaded_layers",
                    "total_layers"):
            value = fields.get(key)
            if isinstance(value, int) and value > 0:
                setattr(f, key, value)
        for key in ("kv_size_mib", "kv_k_mib", "kv_v_mib", "model_size_mib",
                    "prompt_per_second", "eval_per_second"):
            value = fields.get(key)
            if isinstance(value, (int, float)):
                setattr(f, key, float(value))
        for key in ("kv_k_type", "kv_v_type", "file_type", "total_params"):
            value = fields.get(key)
            if isinstance(value, str) and value:
                setattr(f, key, value)

        buffers = fields.get("buffers")
        if isinstance(buffers, list):
            for label, mib in buffers:  # type: ignore[misc]
                f.merge_buffers(str(label), float(mib))

        backend = fields.get("backend")
        if isinstance(backend, tuple):
            label = str(backend[0])
            if label not in f.backends:
                f.backends.append(label)

        device = fields.get("device")
        if isinstance(device, tuple):
            label = f"{device[0]}: {device[1]}".strip().rstrip(")")
            if label not in f.devices:
                f.devices.append(label)

        # An "eval time" line supersedes the prompt-eval throughput reading.
        if "eval_per_second" in fields and "prompt_per_second" in fields:
            pass


def parse_help_banner(line: str) -> Optional[Tuple[str, str]]:
    """Parse lines like ``llama_model_loader: - kv 0: general.name str``."""
    m = re.match(r"^\s*llama_model_loader:\s*-\s*kv\s+(\d+):\s*(\S+)\s*(.*)$", line)
    if not m:
        return None
    return m.group(2), m.group(3).strip()

"""Minimal, read-only GGUF header parser.

This module never loads tensor data.  It walks only the metadata key/value
block and the tensor descriptor table, which both live in the first few
kilobytes to few megabytes of a GGUF file.  Parsing a header is therefore
fast regardless of model size and is safe to call from the UI thread on a
background worker.

Layout (little endian throughout)::

    u32   magic              "GGUF"
    u32   version
    u64   tensor_count
    u64   metadata_kv_count
    []    metadata key/value pairs
    []    tensor descriptors (name, n_dims, dims, ggml_type, offset)
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional, Tuple

MAGIC = 0x46554747  # "GGUF" as a little-endian u32

# ---------------------------------------------------------------- ggml types

#: ggml_type enum -> (display name, bytes per block, elements per block).
#: Types absent from this table are treated as unknown and reported as such.
GGML_TYPE_INFO: Dict[int, Tuple[str, int, int]] = {
    0: ("f32", 4, 1),
    1: ("f16", 2, 1),
    2: ("q4_0", 18, 32),
    3: ("q4_1", 20, 32),
    6: ("q5_0", 22, 32),
    7: ("q5_1", 24, 32),
    8: ("q8_0", 34, 32),
    9: ("q8_1", 40, 32),
    10: ("q2_k", 84, 256),
    11: ("q3_k", 110, 256),
    12: ("q4_k", 144, 256),
    13: ("q5_k", 176, 256),
    14: ("q6_k", 210, 256),
    15: ("q8_k", 292, 256),
    16: ("iq2_xxs", 66, 256),
    17: ("iq2_xs", 74, 256),
    18: ("iq3_xxs", 98, 256),
    19: ("iq1_s", 50, 256),
    20: ("iq4_nl", 18, 32),
    21: ("iq3_s", 110, 256),
    22: ("iq2_s", 82, 256),
    23: ("iq4_xs", 136, 256),
    24: ("i8", 1, 1),
    25: ("i16", 2, 1),
    26: ("i32", 4, 1),
    27: ("i64", 8, 1),
    28: ("f64", 8, 1),
    29: ("iq1_m", 56, 256),
    30: ("bf16", 2, 1),
    31: ("q4_0_4_4", 18, 32),
    32: ("q4_0_4_8", 18, 32),
    33: ("q4_0_8_8", 18, 32),
    34: ("tq1_0", 54, 256),
    35: ("tq2_0", 66, 256),
}

# gguf metadata value type enum
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_SCALAR_STRUCT: Dict[int, Tuple[str, int]] = {
    _GGUF_TYPE_UINT8: ("<B", 1),
    _GGUF_TYPE_INT8: ("<b", 1),
    _GGUF_TYPE_UINT16: ("<H", 2),
    _GGUF_TYPE_INT16: ("<h", 2),
    _GGUF_TYPE_UINT32: ("<I", 4),
    _GGUF_TYPE_INT32: ("<i", 4),
    _GGUF_TYPE_FLOAT32: ("<f", 4),
    _GGUF_TYPE_BOOL: ("<?", 1),
    _GGUF_TYPE_UINT64: ("<Q", 8),
    _GGUF_TYPE_INT64: ("<q", 8),
    _GGUF_TYPE_FLOAT64: ("<d", 8),
}

#: Hard cap on array elements kept in memory (e.g. a 128k-entry token list).
#: Larger arrays are counted and truncated; only their length is used anyway.
_MAX_ARRAY_ELEMENTS = 4096

#: Hard cap on the number of tensor descriptors we retain.  Metadata totals are
#: computed from every descriptor we read, but we only keep this many rows.
_MAX_TENSOR_ROWS = 4096


class GGUFError(ValueError):
    """Raised when a file is not a readable GGUF container."""


@dataclass
class TensorInfo:
    """One entry of the GGUF tensor descriptor table."""

    name: str
    dims: Tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_INFO.get(self.ggml_type, (f"type{self.ggml_type}", 0, 1))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def n_bytes(self) -> Optional[int]:
        """Byte size of the tensor data, or ``None`` for an unknown type."""
        info = GGML_TYPE_INFO.get(self.ggml_type)
        if info is None:
            return None
        _name, block_bytes, block_elems = info
        if block_elems <= 0:
            return None
        n = self.n_elements
        if n % block_elems:
            # Not a whole number of blocks; ggml would not produce this, so be
            # conservative and round up rather than silently under-estimating.
            return (n // block_elems + 1) * block_bytes
        return (n // block_elems) * block_bytes


@dataclass
class GGUFInfo:
    """Parsed GGUF header: metadata, tensor table, and derived totals."""

    path: str
    version: int
    metadata: Dict[str, object] = field(default_factory=dict)
    tensors: List[TensorInfo] = field(default_factory=list)
    tensor_count: int = 0
    tensor_rows_truncated: bool = False
    data_offset: int = 0
    alignment: int = 32
    file_size: int = 0
    error: Optional[str] = None

    # ------------------------------------------------------------- metadata

    def get(self, key: str, default: object = None) -> object:
        return self.metadata.get(key, default)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        v = self.metadata.get(key, default)
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
        return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        v = self.metadata.get(key, default)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return default

    def get_str(self, key: str, default: str = "") -> str:
        v = self.metadata.get(key, default)
        return v if isinstance(v, str) else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self.metadata.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        return default

    def get_array(self, key: str) -> List[object]:
        v = self.metadata.get(key, [])
        return v if isinstance(v, list) else []

    # ----------------------------------------------------------- properties

    @property
    def architecture(self) -> str:
        return self.get_str("general.architecture")

    @property
    def name(self) -> str:
        return self.get_str("general.name") or os.path.basename(self.path)

    @property
    def quantization(self) -> str:
        """Best-effort human readable quantisation name.

        GGUF does not store a canonical quant label, so dominate by bytes over
        the tensor types that carry the weights.  Mixed recipes such as
        ``Q4_K_M`` mix q4_k with a few q6_k tensors, which this reports as
        ``Q4_K/Q6_K``.
        """
        bytes_by_type: Dict[str, int] = {}
        for t in self.tensors:
            if not t.name.endswith(".weight"):
                continue
            n = t.n_bytes
            if n is None:
                continue
            bytes_by_type[t.type_name] = bytes_by_type.get(t.type_name, 0) + n
        if not bytes_by_type:
            return "unknown"
        ranked = sorted(bytes_by_type.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0][0]
        if best.startswith("q") and len(ranked) > 1:
            second = ranked[1][0]
            if second.startswith("q") and bytes_by_type[second] > 0.02 * bytes_by_type[best]:
                return f"{best.upper()}/{second.upper()}"
        return best.upper()

    @property
    def total_tensor_bytes(self) -> int:
        total = 0
        for t in self.tensors:
            b = t.n_bytes
            if b is None:
                return -1
            total += b
        return total

    def arch_get_int(self, suffix: str, default: Optional[int] = None) -> Optional[int]:
        """Read ``<architecture>.<suffix>`` with a fallback to any architecture."""
        arch = self.architecture
        if arch:
            v = self.metadata.get(f"{arch}.{suffix}")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        # Some fine-tunes write the wrong architecture string; fall back to the
        # first matching key so we still produce an estimate.
        for key, v in self.metadata.items():
            if key.endswith("." + suffix) and isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return default

    def arch_get_float(self, suffix: str, default: Optional[float] = None) -> Optional[float]:
        v = self.arch_get_int(suffix)
        return float(v) if v is not None else default

    def arch_get_array(self, suffix: str) -> List[object]:
        arch = self.architecture
        if arch:
            v = self.metadata.get(f"{arch}.{suffix}")
            if isinstance(v, list):
                return v
        for key, v in self.metadata.items():
            if key.endswith("." + suffix) and isinstance(v, list):
                return v
        return []

    def arch_get_bool(self, suffix: str, default: bool = False) -> bool:
        v = self.arch_get_int(suffix)
        return bool(v) if v is not None else default


# ------------------------------------------------------------------ reading


class _Reader:
    """Buffered little-endian reader that tracks the absolute file offset."""

    __slots__ = ("fh", "pos")

    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh
        self.pos = 0

    def read(self, n: int) -> bytes:
        if n < 0:
            raise GGUFError("negative read length")
        buf = self.fh.read(n)
        if len(buf) != n:
            raise GGUFError(f"unexpected end of file at offset {self.pos}")
        self.pos += n
        return buf

    def unpack(self, fmt: str) -> object:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > 64 * 1024 * 1024:
            raise GGUFError(f"implausible string length {n}")
        raw = self.read(n)
        return raw.decode("utf-8", errors="replace")

    def value(self, vtype: int):  # noqa: ANN001 - recursive, type varies
        if vtype in _SCALAR_STRUCT:
            fmt, _size = _SCALAR_STRUCT[vtype]
            return self.unpack(fmt)
        if vtype == _GGUF_TYPE_STRING:
            return self.string()
        if vtype == _GGUF_TYPE_ARRAY:
            elem_type = self.unpack("<I")
            count = self.unpack("<Q")
            if elem_type == _GGUF_TYPE_ARRAY:
                raise GGUFError("nested arrays are not valid GGUF")
            values: List[object] = []
            keep = count <= _MAX_ARRAY_ELEMENTS
            for _ in range(count):
                v = self.value(elem_type)
                if keep:
                    values.append(v)
            return values
        raise GGUFError(f"unknown GGUF metadata type {vtype}")


def load(path: str, *, keep_tensors: bool = True, max_tensors: int = _MAX_TENSOR_ROWS) -> GGUFInfo:
    """Parse the GGUF header at *path*.

    Never raises for a merely malformed file: the returned :class:`GGUFInfo`
    carries an ``error`` string instead, so callers in the UI do not need to
    wrap every call in try/except.  Genuine I/O problems (missing file, no
    permission) still raise ``OSError``.
    """
    info = GGUFInfo(path=path, version=0)
    try:
        info.file_size = os.path.getsize(path)
    except OSError:
        raise
    with open(path, "rb") as fh:
        reader = _Reader(fh)
        try:
            _parse_into(reader, info, keep_tensors=keep_tensors, max_tensors=max_tensors)
        except GGUFError as exc:
            info.error = str(exc)
    return info


def _parse_into(reader: _Reader, info: GGUFInfo, *, keep_tensors: bool, max_tensors: int) -> None:
    magic = reader.unpack("<I")
    if magic != MAGIC:
        raise GGUFError("not a GGUF file (bad magic)")
    info.version = reader.unpack("<I")
    if info.version not in (1, 2, 3):
        raise GGUFError(f"unsupported GGUF version {info.version}")

    tensor_count = reader.unpack("<Q")
    meta_count = reader.unpack("<Q")
    if meta_count > 1_000_000 or tensor_count > 10_000_000:
        raise GGUFError("implausible header counts; file is probably corrupt")
    info.tensor_count = tensor_count

    metadata: Dict[str, object] = {}
    for _ in range(meta_count):
        key = reader.string()
        vtype = reader.unpack("<I")
        metadata[key] = reader.value(vtype)
    info.metadata = metadata

    align = metadata.get("general.alignment")
    if isinstance(align, (int, float)) and int(align) > 0:
        info.alignment = int(align)

    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.unpack("<I")
        if n_dims > 8:
            raise GGUFError(f"implausible tensor rank {n_dims} for {name!r}")
        dims = tuple(reader.unpack("<Q") for _ in range(n_dims))
        ggml_type = reader.unpack("<I")
        offset = reader.unpack("<Q")
        if keep_tensors:
            if len(info.tensors) < max_tensors:
                info.tensors.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))
            else:
                info.tensor_rows_truncated = True

    # The tensor data follows the descriptor table, padded to the alignment.
    raw = reader.pos
    info.data_offset = ((raw + info.alignment - 1) // info.alignment) * info.alignment


def read_info(path: str) -> GGUFInfo:
    """Alias for :func:`load`; kept for readability at call sites."""
    return load(path)


# ------------------------------------------------------------ model geometry


@dataclass
class ModelGeometry:
    """Attention / cache geometry needed to size a KV cache."""

    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    n_ctx_train: int = 0
    head_dim_k: int = 0
    head_dim_v: int = 0
    n_swa: int = 0
    swa_pattern: List[int] = field(default_factory=list)
    n_expert: int = 0
    n_expert_used: int = 0
    #: MLA-style models (DeepSeek) cache a single compressed latent, not K+V.
    kv_lora_rank: int = 0
    rope_dim: int = 0
    attention_type: str = ""
    n_vocab: int = 0
    quant: str = ""
    architecture: str = ""

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0 and self.rope_dim > 0

    @property
    def uses_swa(self) -> bool:
        return self.n_swa > 0

    @property
    def kv_per_token_bytes(self) -> int:
        """Bytes of KV cache consumed by one token in one non-SWA layer."""
        if self.is_mla:
            # Cached per layer: latent (kv_lora_rank) + rope dim, at the cache
            # dtype for the latent part.
            return self.kv_lora_rank + self.rope_dim
        if self.head_dim_k == 0 or self.head_dim_v == 0:
            return 0
        return self.n_head_kv * (self.head_dim_k + self.head_dim_v)


def geometry_from_info(info: GGUFInfo) -> ModelGeometry:
    """Extract cache-relevant geometry from parsed GGUF metadata."""
    g = ModelGeometry()
    g.architecture = info.architecture
    g.quant = info.quantization

    g.n_layer = info.arch_get_int("block_count", 0) or 0
    g.n_embd = info.arch_get_int("embedding_length", 0) or 0
    g.n_head = info.arch_get_int("attention.head_count", 0) or 0
    g.n_head_kv = info.arch_get_int("attention.head_count_kv", 0) or 0
    g.n_ctx_train = info.arch_get_int("context_length", 0) or 0
    g.n_expert = info.arch_get_int("expert_count", 0) or 0
    g.n_expert_used = info.arch_get_int("expert_used_count", 0) or 0
    g.n_vocab = info.get_int("tokenizer.ggml.tokens_count", 0) or 0
    if not g.n_vocab:
        # Most GGUFs omit a token count key; derive it from the token list.
        g.n_vocab = len(info.get_array("tokenizer.ggml.tokens"))

    k_len = info.arch_get_int("attention.key_length", 0) or 0
    v_len = info.arch_get_int("attention.value_length", 0) or 0
    if g.n_head and g.n_embd:
        inferred = g.n_embd // g.n_head
    else:
        inferred = 0
    g.head_dim_k = k_len or inferred
    g.head_dim_v = v_len or inferred

    g.n_swa = info.arch_get_int("attention.sliding_window", 0) or 0
    # Some architectures invert the convention: 0 means "no sliding window".
    pattern = info.arch_get_array("attention.sliding_window_pattern")
    g.swa_pattern = [int(x) for x in pattern if isinstance(x, (int, float))]

    g.kv_lora_rank = info.arch_get_int("attention.kv_lora_rank", 0) or 0
    g.rope_dim = info.arch_get_int("rope.dimension_count", 0) or 0
    g.attention_type = str(info.get("attention.type", "") or "")

    if g.n_head_kv == 0:
        # No GQA metadata: assume one KV head per query head.
        g.n_head_kv = g.n_head
    if g.n_ctx_train == 0:
        g.n_ctx_train = 4096
    return g


def is_mmproj(path: str) -> bool:
    """True when *path* looks like a multimodal projector GGUF."""
    base = os.path.basename(path).lower()
    return "mmproj" in base


def parse_shard_name(name: str) -> Optional[Tuple[int, int]]:
    """Extract ``(index, total)`` from an ``...-00001-of-00003.gguf`` name."""
    import re

    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

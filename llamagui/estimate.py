"""Static VRAM / context estimation from GGUF metadata.

Nothing here loads a model or talks to the GPU.  The numbers are derived only
from the parsed GGUF header plus the arguments the user has chosen, which makes
them available before a launch is attempted.  They are deliberately labelled as
estimates: the weights term is exact (sum of tensor byte sizes plus a small
per-architecture overhead), the KV term is exact for the sampled geometry, and
the compute-buffer term is a calibrated heuristic.

The estimator is used in two directions:

* *forward* - "given this context and these cache types, how much VRAM?"
* *inverse* - "given this much free VRAM, what is the largest context?"

``llama.cpp``'s own ``--fit`` option remains the authority at run time; this
module exists to catch an obviously impossible configuration before the user
waits several minutes for a load to fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .gguf import GGUFInfo, ModelGeometry

MiB = 1024 * 1024
GiB = 1024 * MiB

#: Bytes per cache element for each accepted ``-ctk`` / ``-ctv`` value.
#: K-cache quantisation stores the block scale per 32 elements (hence 34/32 for
#: q8_0 and 18/32 for q4_0); V-cache quantisation has no scale per block.
KV_DTYPE_BYTES: Dict[str, Tuple[float, float]] = {
    # value: (bytes per element for K, bytes per element for V)
    "f32": (4.0, 4.0),
    "f16": (2.0, 2.0),
    "bf16": (2.0, 2.0),
    "q8_0": (34.0 / 32.0, 34.0 / 32.0),
    "q5_1": (24.0 / 32.0, 24.0 / 32.0),
    "q5_0": (22.0 / 32.0, 22.0 / 32.0),
    "q4_1": (20.0 / 32.0, 20.0 / 32.0),
    "q4_0": (18.0 / 32.0, 18.0 / 32.0),
    "iq4_nl": (18.0 / 32.0, 18.0 / 32.0),
}

DEFAULT_CACHE_TYPE_K = "f16"
DEFAULT_CACHE_TYPE_V = "f16"

#: Per-architecture weight overhead on top of the raw tensor bytes.  CUDA
#: keeps extra copies for some operations and pads allocations.
_WEIGHT_OVERHEAD_DEFAULT = 0.03

#: Compute-buffer heuristic coefficients.  The buffer scales with the physical
#: micro-batch and, for the output projection, with the vocabulary size.
_COMPUTE_BASE_MIB = 96.0
_COMPUTE_PER_UBATCH_TOK_MIB = 0.35
_LOGITS_MIB_PER_VOCAB_PER_UBATCH = 4.0 / (1024 * 1024)  # f32 logits


@dataclass
class Breakdown:
    """A single labelled VRAM component."""

    label: str
    bytes: int
    note: str = ""
    heuristic: bool = False

    @property
    def mib(self) -> float:
        return self.bytes / MiB

    @property
    def gib(self) -> float:
        return self.bytes / GiB


@dataclass
class Estimate:
    """Result of a static estimation pass."""

    geometry: ModelGeometry
    n_ctx: int = 0
    n_ctx_train: int = 0
    n_parallel: int = 1
    kv_unified: bool = True
    ctk: str = DEFAULT_CACHE_TYPE_K
    ctv: str = DEFAULT_CACHE_TYPE_V

    weights_bytes: int = 0
    kv_bytes: int = 0
    compute_bytes: int = 0
    device_total_bytes: int = 0
    device_free_bytes: int = 0

    #: Context length at which the configuration is expected to just fit.
    max_ctx_fit: int = 0
    #: Context length that fits if the compute buffer is allowed to be tight.
    max_ctx_tight: int = 0

    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes + self.compute_bytes

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GiB

    @property
    def total_mib(self) -> float:
        return self.total_bytes / MiB

    @property
    def free_gib(self) -> float:
        return self.device_free_bytes / GiB

    @property
    def total_vram_gib(self) -> float:
        return self.device_total_bytes / GiB

    @property
    def headroom_bytes(self) -> int:
        return self.device_free_bytes - self.total_bytes

    @property
    def headroom_gib(self) -> float:
        return self.headroom_bytes / GiB

    @property
    def fits(self) -> bool:
        return self.device_free_bytes > 0 and self.total_bytes <= self.device_free_bytes

    @property
    def kv_per_token_bytes(self) -> int:
        """KV bytes added by one extra token, given the current cache types."""
        if self.n_ctx <= 0 or self.kv_bytes <= 0:
            return 0
        n_seq = 1 if self.kv_unified else max(1, self.n_parallel)
        depth = max(1, self.n_ctx * n_seq)
        return int(self.kv_bytes / depth)

    def breakdown(self) -> List[Breakdown]:
        items = [
            Breakdown("模型权重", self.weights_bytes, "GGUF 张量字节总和 × 架构系数"),
            Breakdown(
                "KV cache",
                self.kv_bytes,
                f"K={self.ctk} V={self.ctv}, ctx={self.n_ctx}"
                + (f", {self.n_parallel} 并发槽" if self.n_parallel > 1 else ""),
            ),
            Breakdown("计算缓冲/开销", self.compute_bytes, "按 ubatch 与词表大小估算", heuristic=True),
        ]
        if self.device_free_bytes:
            items.append(
                Breakdown("空闲显存", self.device_free_bytes, "启动时刻实测", heuristic=True)
            )
        return items


# ------------------------------------------------------------------ helpers


def kv_bytes_per_token_per_layer(geometry: ModelGeometry, ctk: str, ctv: str) -> float:
    """Bytes one token occupies in one full-attention layer's KV cache."""
    k_elem, v_elem = KV_DTYPE_BYTES.get(ctk, KV_DTYPE_BYTES[DEFAULT_CACHE_TYPE_K])[0], KV_DTYPE_BYTES.get(ctv, KV_DTYPE_BYTES[DEFAULT_CACHE_TYPE_V])[1]
    if geometry.is_mla:
        # MLA caches a single compressed latent plus the rope component.
        return geometry.kv_lora_rank * k_elem + geometry.rope_dim * v_elem
    if geometry.head_dim_k <= 0 or geometry.head_dim_v <= 0:
        return 0.0
    return geometry.n_head_kv * (geometry.head_dim_k * k_elem + geometry.head_dim_v * v_elem)


def layer_depths(geometry: ModelGeometry, n_ctx: int, swa_full: bool = False) -> List[int]:
    """Effective KV depth for every layer, honouring sliding-window patterns.

    Returns a list of length ``n_layer``.  A sliding-window layer never needs
    more than ``n_swa`` positions, so enabling SWA can cut the cache
    dramatically for models that use it on every layer.
    """
    n_layer = max(0, geometry.n_layer)
    depths = [n_ctx] * n_layer
    if not geometry.uses_swa or swa_full or n_ctx <= geometry.n_swa:
        return depths

    pattern = geometry.swa_pattern
    if pattern:
        # ``swa_pattern`` repeats with its own period across the layer stack.
        period = len(pattern)
        for i in range(n_layer):
            if pattern[i % period]:
                depths[i] = min(n_ctx, geometry.n_swa)
    else:
        # No explicit pattern: llama.cpp applies the window to every layer
        # except the final one for the architectures that expose n_swa.
        for i in range(n_layer):
            if i < n_layer - 1:
                depths[i] = min(n_ctx, geometry.n_swa)
    return depths


def _kv_layer_slots(
    geometry: ModelGeometry,
    n_ctx: int,
    *,
    n_parallel: int = 1,
    kv_unified: bool = True,
    swa_full: bool = False,
) -> float:
    """Sum of per-layer KV depths for a *n_ctx*-token context.

    This is the single source of truth shared by the forward query
    (:func:`kv_cache_bytes`) and the inverse query
    (:func:`_kv_bytes_per_token_total`), so the two can never drift apart.
    """
    if n_ctx <= 0 or geometry.n_layer <= 0:
        return 0
    n_seq = 1 if kv_unified else max(1, n_parallel)
    depths = layer_depths(geometry, n_ctx * n_seq, swa_full=swa_full)
    return sum(depths)


def kv_cache_bytes(
    geometry: ModelGeometry,
    n_ctx: int,
    ctk: str = DEFAULT_CACHE_TYPE_K,
    ctv: str = DEFAULT_CACHE_TYPE_V,
    *,
    n_parallel: int = 1,
    kv_unified: bool = True,
    swa_full: bool = False,
    kv_offloaded: bool = True,
) -> int:
    """Total KV cache size in bytes for the given context and cache types.

    ``kv_offloaded=False`` (``--no-kv-offload``) means the cache lives in host
    RAM rather than VRAM, so callers should exclude this from VRAM totals.
    """
    if not kv_offloaded:
        return 0
    per_token_layer = kv_bytes_per_token_per_layer(geometry, ctk, ctv)
    if per_token_layer <= 0:
        return 0
    slots = _kv_layer_slots(
        geometry, n_ctx, n_parallel=n_parallel, kv_unified=kv_unified, swa_full=swa_full
    )
    return int(slots * per_token_layer)


def compute_buffer_bytes(
    geometry: ModelGeometry,
    ubatch: int = 512,
    kv_type_v: str = DEFAULT_CACHE_TYPE_V,
) -> int:
    """Heuristic size of the compute / activation buffers.

    Calibrated from observed llama.cpp CUDA runs: the dominant terms are the
    f32 logits tensor (``ubatch × vocab``) and the attention / FFN activations
    that scale with the micro-batch size.
    """
    ub = max(1, ubatch)
    vocab = geometry.n_vocab or 32000
    logits = int(ub * vocab * _LOGITS_MIB_PER_VOCAB_PER_UBATCH * MiB)
    activations = int((_COMPUTE_BASE_MIB + _COMPUTE_PER_UBATCH_TOK_MIB * ub) * MiB)
    # Quantised V cache needs a dequantisation staging buffer per layer chunk.
    if kv_type_v in ("q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "q8_0"):
        activations = int(activations * 1.15)
    return logits + activations


def weight_overhead_factor(architecture: str) -> float:
    arch = (architecture or "").lower()
    if arch in ("deepseek2", "deepseek3", "glm4moe", "qwen3moe", "kimi_k2"):
        return 0.05
    if arch in ("gpt-oss",):
        return 0.04
    return _WEIGHT_OVERHEAD_DEFAULT


# ----------------------------------------------------------------- estimate


def estimate(
    info: GGUFInfo,
    *,
    n_ctx: int = 0,
    ctk: str = DEFAULT_CACHE_TYPE_K,
    ctv: str = DEFAULT_CACHE_TYPE_V,
    n_parallel: int = 1,
    kv_unified: bool = True,
    swa_full: bool = False,
    kv_offloaded: bool = True,
    ubatch: int = 512,
    device_total_bytes: int = 0,
    device_free_bytes: int = 0,
    fit_margin_bytes: int = 1024 * MiB,
    gpu_layers: Optional[str] = None,
    n_layer_total: Optional[int] = None,
) -> Estimate:
    """Produce a static VRAM estimate for one candidate configuration."""
    geometry = info_geometry(info)
    est = Estimate(
        geometry=geometry,
        n_ctx=n_ctx,
        n_ctx_train=geometry.n_ctx_train,
        n_parallel=max(1, n_parallel),
        kv_unified=kv_unified,
        ctk=ctk,
        ctv=ctv,
        device_total_bytes=device_total_bytes,
        device_free_bytes=device_free_bytes,
    )

    if info.error:
        est.warnings.append(f"GGUF 头解析失败：{info.error}")
        return est

    raw_weights = info.total_tensor_bytes
    if raw_weights < 0:
        est.warnings.append("存在未知张量类型，权重体积无法精确统计")
        raw_weights = max(0, info.file_size)
    overhead = weight_overhead_factor(geometry.architecture)
    est.weights_bytes = int(raw_weights * (1.0 + overhead))

    total_layers = geometry.n_layer or (n_layer_total or 0)
    offload_layers = _resolve_offload_layers(gpu_layers, total_layers)
    if total_layers > 0 and offload_layers < total_layers:
        frac = offload_layers / total_layers
        est.weights_bytes = int(est.weights_bytes * frac)
        est.notes.append(
            f"仅 {offload_layers}/{total_layers} 层卸载到 GPU，权重按比例折算"
            f"（未卸载层由 CPU 承担，速度会显著下降）"
        )

    est.kv_bytes = kv_cache_bytes(
        geometry,
        n_ctx,
        ctk,
        ctv,
        n_parallel=n_parallel,
        kv_unified=kv_unified,
        swa_full=swa_full,
        kv_offloaded=kv_offloaded,
    )
    est.compute_bytes = compute_buffer_bytes(geometry, ubatch=ubatch, kv_type_v=ctv)

    if n_ctx <= 0:
        est.notes.append("未指定上下文：llama.cpp 将采用模型自带值，估算以训练上下文为参考")
    elif geometry.n_ctx_train and n_ctx > geometry.n_ctx_train:
        est.warnings.append(
            f"请求上下文 {n_ctx} 超过模型训练上下文 {geometry.n_ctx_train}，"
            f"需要 RoPE/YaRN 缩放，超出太多会导致输出质量明显下降"
        )

    if geometry.uses_swa and not swa_full:
        est.notes.append(
            f"模型使用滑动窗口注意力（窗口 {geometry.n_swa}），KV cache 已按窗口折算"
        )

    if not kv_offloaded:
        est.notes.append("--no-kv-offload：KV cache 位于内存而非显存，已从显存估算中排除")

    _compute_context_limits(est, geometry, fit_margin_bytes, swa_full=swa_full)
    return est


def info_geometry(info: GGUFInfo) -> ModelGeometry:
    from .gguf import geometry_from_info

    return geometry_from_info(info)


def _resolve_offload_layers(gpu_layers: Optional[str], total_layers: int) -> int:
    if total_layers <= 0:
        return 0
    if gpu_layers is None:
        return total_layers
    gl = str(gpu_layers).strip().lower()
    if gl in ("", "auto", "all", "-1"):
        return total_layers
    try:
        n = int(gl)
    except ValueError:
        return total_layers
    if n < 0:
        return total_layers
    return min(n, total_layers)


def _compute_context_limits(
    est: Estimate,
    geometry: ModelGeometry,
    fit_margin_bytes: int,
    *,
    swa_full: bool = False,
) -> None:
    """Fill in the largest context that is expected to fit.

    Sweeps a candidate ladder and keeps the largest value whose *exact* KV cost
    fits, rather than dividing by a per-token slope.  The exact form matters for
    sliding-window models, where KV growth is linear only until the window fills
    and is flat afterwards.
    """
    if not est.device_free_bytes:
        return
    fixed = est.weights_bytes + est.compute_bytes
    budget = est.device_free_bytes - fit_margin_bytes
    if budget <= fixed:
        est.warnings.append(
            "权重加计算缓冲已经超过可用显存（考虑预留 margin 后无空间给 KV cache）"
        )
        est.max_ctx_fit = 0
        est.max_ctx_tight = 0
        return

    def kv_at(ctx: int) -> int:
        return kv_cache_bytes(
            geometry,
            ctx,
            est.ctk,
            est.ctv,
            n_parallel=est.n_parallel,
            kv_unified=est.kv_unified,
            swa_full=swa_full,
            kv_offloaded=True,
        )

    cap = geometry.n_ctx_train or 0
    est.max_ctx_fit = _largest_fitting(lambda c: fixed + kv_at(c) <= budget, cap)
    est.max_ctx_tight = _largest_fitting(
        lambda c: fixed + kv_at(c) <= est.device_free_bytes, cap
    )


#: Context candidates used when searching for the largest value that fits.
#: Ascending, so the sweep can stop at the first miss.
CTX_CANDIDATES: Tuple[int, ...] = (
    1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 40960,
    49152, 65536, 81920, 98304, 131072, 163840, 196608, 262144, 327680,
    393216, 524288, 786432, 1048576, 2097152,
)


def _largest_fitting(predicate, cap: int) -> int:  # noqa: ANN001
    """Largest candidate satisfying ``predicate(ctx)``, bounded by *cap*."""
    best = 0
    for ctx in CTX_CANDIDATES:
        if cap and ctx > cap:
            break
        if predicate(ctx):
            best = ctx
        else:
            # KV cost is monotonically non-decreasing in ctx, so the first miss
            # implies every later candidate also misses.
            break
    return best


def _kv_bytes_per_token_total(
    est: Estimate, geometry: ModelGeometry, *, swa_full: bool = False
) -> float:
    """KV growth per additional context token in the linear (pre-window) regime."""
    slots = _kv_layer_slots(
        geometry,
        1,
        n_parallel=est.n_parallel,
        kv_unified=est.kv_unified,
        swa_full=swa_full,
    )
    return slots * kv_bytes_per_token_per_layer(geometry, est.ctk, est.ctv)


def suggest_context(
    info: GGUFInfo,
    *,
    device_free_bytes: int,
    ctk: str = DEFAULT_CACHE_TYPE_K,
    ctv: str = DEFAULT_CACHE_TYPE_V,
    ubatch: int = 512,
    n_parallel: int = 1,
    kv_unified: bool = True,
    swa_full: bool = False,
    fit_margin_bytes: int = 1024 * MiB,
    gpu_layers: Optional[str] = None,
) -> int:
    """Largest context size that is expected to fit in free VRAM.

    Thin wrapper over :func:`estimate`: the sweep lives in
    :func:`_largest_fitting` so there is exactly one implementation of the
    "what fits" search.
    """
    if device_free_bytes <= 0:
        return 0
    est = estimate(
        info,
        n_ctx=0,
        ctk=ctk,
        ctv=ctv,
        n_parallel=n_parallel,
        kv_unified=kv_unified,
        swa_full=swa_full,
        ubatch=ubatch,
        device_free_bytes=device_free_bytes,
        fit_margin_bytes=fit_margin_bytes,
        gpu_layers=gpu_layers,
    )
    return est.max_ctx_fit


# ------------------------------------------------------------- presentation


def human_bytes(n: float) -> str:
    """Format a byte count using binary units, e.g. ``13.42 GiB``."""
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            if unit == "B":
                return f"{sign}{int(n)} B"
            return f"{sign}{n:.2f} {unit}"
        n /= 1024
    return f"{sign}{n:.2f} TiB"

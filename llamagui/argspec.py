"""Curated metadata for the arguments the GUI exposes.

The installed binaries' own ``--help`` output (see :mod:`helpparse`) is the
authority on *which* options exist.  This module adds what help text cannot
express directly: which tab an option belongs to, what widget to draw, what the
default is for the purpose of "only emit non-defaults", and which cross-argument
rules apply.

The invariant enforced by the test suite is that every :class:`ArgSpec` here
uses a canonical long flag that :mod:`helpparse` can also find in the local
help dump.  If a future build renames or removes an option, the UI greys it out
instead of emitting an argument the binary would reject.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .estimate import DEFAULT_CACHE_TYPE_K, DEFAULT_CACHE_TYPE_V


class Kind(enum.Enum):
    """How an argument is edited and serialised."""

    #: Bare toggle.  ``True`` emits the flag, ``False`` emits ``spec.neg``.
    BOOL = "bool"
    #: Optional value from :attr:`ArgSpec.choices`; ``None`` emits nothing.
    #: Used for ``--flash-attn [on|off|auto]`` and friends.
    TRISTATE = "tristate"
    #: Single choice from :attr:`ArgSpec.choices`, always emitted when set.
    CHOICE = "choice"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    #: String with newlines (grammars, JSON schema); passed as one argument.
    TEXT = "text"
    #: Comma separated list, kept verbatim: ``dev0,dev1`` / ``3,1``.
    CSV = "csv"
    PATH = "path"
    #: Repeatable option; each list entry becomes ``flag value``.
    REPEAT = "repeat"


class Group(enum.Enum):
    """Notebook tab an argument belongs to."""

    MODEL = "模型"
    CONTEXT = "上下文与批处理"
    HARDWARE = "显存与硬件"
    SAMPLING = "采样与生成"
    SPECULATIVE = "投机解码"
    SERVER = "服务器与 API"
    LOGGING = "日志与安全"
    BENCH = "基准测试"


@dataclass(frozen=True)
class ArgSpec:
    """One user-facing argument."""

    flag: str
    kind: Kind
    group: Group
    label: str
    default: object = None
    choices: Tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""
    env: str = ""
    help: str = ""
    subgroup: str = ""
    #: Long flag emitted when a BOOL is set to ``False``.
    neg: str = ""
    #: Section within the tab, used to lay out labelled frames.
    section: str = ""
    #: Options with a security or stability impact; rendered with a warning.
    danger: bool = False
    #: Documented as experimental upstream.
    experimental: bool = False
    #: Hide unless the user enables advanced mode.
    advanced: bool = False
    #: Shown in the "quick preset" shortlist.
    common: bool = False
    #: Holds a credential.  Rendered masked with a reveal toggle so it cannot
    #: leak through a screenshot or over someone's shoulder.
    secret: bool = False

    # ---------------------------------------------------------------- checks

    def coerce(self, value: object) -> object:
        """Convert *value* to this spec's Python type, or raise ``ValueError``."""
        if value is None or value == "":
            return None
        if self.kind is Kind.BOOL:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True
            if text in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"{self.flag}: 需要布尔值，收到 {value!r}")
        if self.kind is Kind.INT:
            try:
                n = int(str(value).strip())
            except ValueError as exc:
                raise ValueError(f"{self.flag}: 需要整数，收到 {value!r}") from exc
            self.check_range(n)
            return n
        if self.kind is Kind.FLOAT:
            try:
                f = float(str(value).strip())
            except ValueError as exc:
                raise ValueError(f"{self.flag}: 需要数字，收到 {value!r}") from exc
            self.check_range(f)
            return f
        if self.kind in (Kind.CHOICE, Kind.TRISTATE):
            text = str(value).strip()
            if not text:
                return None
            if self.choices and text not in self.choices:
                raise ValueError(
                    f"{self.flag}: 取值必须是 {', '.join(self.choices)} 之一，收到 {text!r}"
                )
            return text
        return str(value)

    def check_range(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.flag}: 不能小于 {self.minimum}（收到 {value}）")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.flag}: 不能大于 {self.maximum}（收到 {value}）")

    def is_default(self, value: object) -> bool:
        """True when *value* equals this spec's default (so it can be omitted)."""
        if value is None:
            return True
        if self.default is None:
            return False
        if self.kind in (Kind.INT, Kind.FLOAT):
            try:
                return float(value) == float(self.default)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False
        if self.kind is Kind.BOOL:
            return bool(value) is bool(self.default)
        return str(value) == str(self.default)

    def serialize(self, value: object) -> List[str]:
        """Turn *value* into command line tokens for this option.

        Returns an empty list when the value should be omitted entirely, which
        is the mechanism that keeps ``llama.cpp`` defaults (and therefore
        ``--fit``) in charge for anything the user did not explicitly set.
        """
        if value is None:
            return []
        kind = self.kind

        if kind is Kind.BOOL:
            if bool(value) is True:
                return [self.flag]
            if bool(value) is False and self.neg:
                return [self.neg]
            return []

        if kind in (Kind.CHOICE, Kind.TRISTATE):
            text = str(value)
            if not text:
                return []
            return [self.flag, text]

        if kind is Kind.REPEAT:
            if isinstance(value, (list, tuple)):
                out: List[str] = []
                for item in value:
                    out.extend([self.flag, str(item)])
                return out
            return [self.flag, str(value)]

        text = str(value)
        if text == "":
            return []
        if kind is Kind.INT:
            try:
                return [self.flag, str(int(text))]
            except ValueError:
                return []
        if kind is Kind.FLOAT:
            try:
                f = float(text)
            except ValueError:
                return []
            # Avoid emitting ``0.8000000001`` style noise after float round-trips.
            if f == int(f) and abs(f) < 1e15:
                return [self.flag, str(int(f))]
            return [self.flag, repr(f)]
        return [self.flag, text]


# --------------------------------------------------------------------- table


def _specs() -> List[ArgSpec]:
    """Build the argument table.  Kept as a function for readability."""
    s: List[ArgSpec] = []
    add = s.append

    M = Group.MODEL
    C = Group.CONTEXT
    H = Group.HARDWARE
    S = Group.SAMPLING
    P = Group.SPECULATIVE
    V = Group.SERVER
    L = Group.LOGGING
    B = Group.BENCH

    # =================================================================== model
    add(ArgSpec("--model", Kind.PATH, M, "模型文件 (.gguf)", section="模型来源", common=True,
                help="本地 GGUF 路径。留空切换到 HF 仓库或路由模式。",
                env="LLAMA_ARG_MODEL"))
    add(ArgSpec("--model-url", Kind.STR, M, "模型下载地址", section="模型来源", advanced=True,
                env="LLAMA_ARG_MODEL_URL"))
    add(ArgSpec("--hf-repo", Kind.STR, M, "HuggingFace 仓库", section="模型来源", common=True,
                help="格式 user/model[:quant]，例如 ggml-org/gemma-3-4b-it-GGUF:Q4_K_M。会自动下载。",
                env="LLAMA_ARG_HF_REPO"))
    add(ArgSpec("--hf-file", Kind.STR, M, "HF 指定文件", section="模型来源", advanced=True,
                env="LLAMA_ARG_HF_FILE"))
    add(ArgSpec("--hf-token", Kind.STR, M, "HF Token", section="模型来源", advanced=True,
                env="HF_TOKEN", secret=True))
    add(ArgSpec("--docker-repo", Kind.STR, M, "Docker 模型仓库", section="模型来源", advanced=True,
                env="LLAMA_ARG_DOCKER_REPO"))
    add(ArgSpec("--offline", Kind.BOOL, M, "离线模式（禁止联网）", default=False,
                section="模型来源", advanced=True, env="LLAMA_ARG_OFFLINE"))
    add(ArgSpec("--mmproj-auto", Kind.BOOL, M, "自动下载 mmproj", default=True,
                neg="--no-mmproj-auto", section="多模态", env="LLAMA_ARG_MMPROJ_AUTO",
                help="使用 -hf 时自动获取多模态投影文件。关闭后变成 --no-mmproj-auto。"))
    add(ArgSpec("--mmproj", Kind.PATH, M, "多模态投影文件 mmproj", section="多模态",
                env="LLAMA_ARG_MMPROJ"))
    add(ArgSpec("--mmproj-url", Kind.STR, M, "mmproj 下载地址", section="多模态", advanced=True,
                env="LLAMA_ARG_MMPROJ_URL"))
    add(ArgSpec("--mmproj-offload", Kind.BOOL, M, "mmproj 卸载到 GPU", default=True,
                neg="--no-mmproj-offload", section="多模态"))
    add(ArgSpec("--mmproj-device", Kind.STR, M, "mmproj 设备", section="多模态", advanced=True,
                env="MTMD_BACKEND_DEVICE"))
    add(ArgSpec("--image-min-tokens", Kind.INT, M, "每张图最少 token", minimum=0,
                section="多模态", advanced=True, env="LLAMA_ARG_IMAGE_MIN_TOKENS"))
    add(ArgSpec("--image-max-tokens", Kind.INT, M, "每张图最多 token", minimum=0,
                section="多模态", advanced=True, env="LLAMA_ARG_IMAGE_MAX_TOKENS"))
    add(ArgSpec("--mtmd-batch-max-tokens", Kind.INT, M, "图像批处理上限", minimum=0,
                section="多模态", advanced=True, env="LLAMA_ARG_MTMD_BATCH_MAX_TOKENS"))
    add(ArgSpec("--video-fps", Kind.FLOAT, M, "视频帧率", section="多模态", advanced=True,
                env="LLAMA_ARG_VIDEO_FPS"))
    add(ArgSpec("--video-timestamp-interval", Kind.INT, M, "视频时间戳间隔(ms)", minimum=0,
                section="多模态", advanced=True, env="LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL"))
    add(ArgSpec("--video-ffmpeg-dir", Kind.PATH, M, "ffmpeg 目录", section="多模态", advanced=True,
                env="LLAMA_ARG_VIDEO_FFMPEG_DIR"))
    add(ArgSpec("--media-path", Kind.PATH, M, "本地素材目录", section="多模态", advanced=True,
                help="启用 file:// 方式向 API 传入本地图片/音频/视频。"))
    add(ArgSpec("--alias", Kind.CSV, M, "模型别名（API model 字段）", section="对话模板",
                env="LLAMA_ARG_ALIAS"))
    add(ArgSpec("--tags", Kind.CSV, M, "模型标签", section="对话模板", advanced=True,
                env="LLAMA_ARG_TAGS"))
    add(ArgSpec("--jinja", Kind.BOOL, M, "使用 Jinja 对话模板", default=True, neg="--no-jinja",
                section="对话模板", common=True, env="LLAMA_ARG_JINJA",
                help="工具调用（function calling）依赖它，建议保持开启。"))
    add(ArgSpec("--chat-template", Kind.TEXT, M, "自定义对话模板", section="对话模板",
                help="直接写 Jinja 模板文本。"))
    add(ArgSpec("--chat-template-file", Kind.PATH, M, "对话模板文件", section="对话模板"))
    add(ArgSpec("--chat-template-kwargs", Kind.STR, M, "模板额外参数 (JSON)", section="对话模板",
                help='例如 {"enable_thinking": false}'))
    add(ArgSpec("--skip-chat-parsing", Kind.BOOL, M, "跳过对话解析", default=False,
                section="对话模板", advanced=True))
    add(ArgSpec("--prefill-assistant", Kind.BOOL, M, "预填充 assistant 消息", default=False,
                section="对话模板", advanced=True))
    add(ArgSpec("--reasoning", Kind.CHOICE, M, "推理/思考模式", choices=("on", "off", "auto"),
                section="推理", env="LLAMA_ARG_REASONING"))
    add(ArgSpec("--reasoning-format", Kind.STR, M, "推理输出去向", section="推理",
                help="none = 原样输出原始文本。"))
    add(ArgSpec("--reasoning-effort", Kind.STR, M, "推理强度", section="推理", advanced=True))
    add(ArgSpec("--reasoning-budget", Kind.INT, M, "思考 token 预算", section="推理",
                help="-1 不限，0 立即结束。"))
    add(ArgSpec("--reasoning-budget-message", Kind.STR, M, "预算耗尽提示", section="推理", advanced=True))
    add(ArgSpec("--reasoning-preserve", Kind.BOOL, M, "保留历史推理内容", default=False,
                neg="--no-reasoning-preserve", section="推理", advanced=True))
    add(ArgSpec("--lora", Kind.REPEAT, M, "LoRA 适配器", section="LoRA 与控制向量"))
    add(ArgSpec("--lora-scaled", Kind.REPEAT, M, "LoRA + 缩放", section="LoRA 与控制向量",
                help="格式 FNAME:SCALE"))
    add(ArgSpec("--lora-init-without-apply", Kind.BOOL, M, "加载但不启用 LoRA", default=False,
                section="LoRA 与控制向量", advanced=True))
    add(ArgSpec("--control-vector", Kind.REPEAT, M, "控制向量", section="LoRA 与控制向量"))
    add(ArgSpec("--control-vector-scaled", Kind.REPEAT, M, "控制向量 + 缩放",
                section="LoRA 与控制向量"))
    add(ArgSpec("--control-vector-layer-range", Kind.REPEAT, M, "控制向量层范围",
                section="LoRA 与控制向量", advanced=True, help="两个值：起始 结束（含）"))
    add(ArgSpec("--models-dir", Kind.PATH, M, "路由模式：模型目录", section="路由模式",
                env="LLAMA_ARG_MODELS_DIR",
                help="不指定 --model 时启用：llama-server 变为多模型路由，按需加载。"))
    add(ArgSpec("--models-preset", Kind.PATH, M, "路由模式：预设 INI", section="路由模式",
                env="LLAMA_ARG_MODELS_PRESET"))
    add(ArgSpec("--models-max", Kind.INT, M, "同时加载模型上限", minimum=1, section="路由模式",
                env="LLAMA_ARG_MODELS_MAX"))
    add(ArgSpec("--models-autoload", Kind.BOOL, M, "路由模式：按需自动加载", default=True,
                neg="--no-models-autoload", section="路由模式"))

    # ================================================================= context
    add(ArgSpec("--ctx-size", Kind.INT, C, "上下文长度 -c", minimum=0, section="上下文",
                common=True, env="LLAMA_ARG_CTX_SIZE",
                help="0 = 采用模型自带值（可能非常大，先看估算卡片）。"))
    add(ArgSpec("--predict", Kind.INT, C, "最大生成 token", default=-1, section="上下文",
                env="LLAMA_ARG_N_PREDICT", help="-1 = 不限。"))
    add(ArgSpec("--parallel", Kind.INT, C, "并发槽位 -np", minimum=1, maximum=256,
                section="上下文", env="LLAMA_ARG_N_PARALLEL",
                help="-1 = 自动。每个槽位共享或各自占用上下文，注意 KV 总量。"))
    add(ArgSpec("--batch-size", Kind.INT, C, "逻辑批 -b", default=2048, minimum=1,
                section="批处理", env="LLAMA_ARG_BATCH"))
    add(ArgSpec("--ubatch-size", Kind.INT, C, "物理批 -ub", default=512, minimum=1,
                section="批处理", common=True, env="LLAMA_ARG_UBATCH",
                help="必须是 32 的倍数，直接影响显存峰值与计算缓冲。"))
    add(ArgSpec("--keep", Kind.INT, C, "保留初始 prompt token", default=0, section="批处理",
                help="-1 = 全部保留。"))
    add(ArgSpec("--cont-batching", Kind.BOOL, C, "连续批处理", default=True, neg="--no-cont-batching",
                section="批处理", advanced=True, env="LLAMA_ARG_CONT_BATCHING"))
    add(ArgSpec("--kv-unified", Kind.BOOL, C, "统一 KV 缓冲", default=True, neg="--no-kv-unified",
                section="批处理", advanced=True, env="LLAMA_ARG_KV_UNIFIED",
                help="所有并发槽共享一个 KV 池；关闭后每槽独立分配 -c。"))
    add(ArgSpec("--kv-unified-per-slot", Kind.INT, C, "每槽上下文上限", minimum=0,
                section="批处理", advanced=True, env="LLAMA_ARG_KV_UNIFIED_PER_SLOT"))
    add(ArgSpec("--context-shift", Kind.BOOL, C, "上下文滑动（无限生成）", default=False,
                neg="--no-context-shift", section="批处理", advanced=True,
                env="LLAMA_ARG_CONTEXT_SHIFT"))
    add(ArgSpec("--swa-full", Kind.BOOL, C, "全尺寸 SWA cache", default=False, section="批处理",
                advanced=True, env="LLAMA_ARG_SWA_FULL",
                help="滑动窗口模型默认只缓存窗口大小；开启后按完整上下文分配，显著增加显存。"))
    add(ArgSpec("--ctx-checkpoints", Kind.INT, C, "上下文检查点数量", minimum=0,
                section="批处理", advanced=True, env="LLAMA_ARG_CTX_CHECKPOINTS"))
    add(ArgSpec("--checkpoint-min-step", Kind.INT, C, "检查点最小间隔", minimum=0,
                section="批处理", advanced=True, env="LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT"))
    add(ArgSpec("--threads", Kind.INT, C, "生成线程 -t", minimum=-1, maximum=512, section="CPU",
                common=True, env="LLAMA_ARG_THREADS",
                help="-1 = 自动。纯 GPU 推理通常设为物理核心数即可，过多线程反而变慢。"))
    add(ArgSpec("--threads-batch", Kind.INT, C, "批处理线程 -tb", minimum=-1, maximum=512,
                section="CPU"))
    add(ArgSpec("--cpu-mask", Kind.STR, C, "CPU 亲和掩码 -C", section="CPU", advanced=True))
    add(ArgSpec("--cpu-range", Kind.STR, C, "CPU 亲和范围 -Cr", section="CPU", advanced=True,
                help="形如 0-7"))
    add(ArgSpec("--cpu-strict", Kind.TRISTATE, C, "严格 CPU 绑定", choices=("0", "1"),
                section="CPU", advanced=True))
    add(ArgSpec("--prio", Kind.INT, C, "进程优先级", minimum=-1, maximum=3, section="CPU",
                advanced=True, help="-1 低 / 0 普通 / 1 中 / 2 高 / 3 实时(谨慎)"))
    add(ArgSpec("--poll", Kind.INT, C, "轮询级别", minimum=0, maximum=100, default=50,
                section="CPU", advanced=True))
    add(ArgSpec("--numa", Kind.CHOICE, C, "NUMA 策略", choices=("distribute", "isolate", "numactl"),
                section="CPU", advanced=True, env="LLAMA_ARG_NUMA",
                help="Windows 上通常无用，仅在多路 NUMA 服务器上有意义。"))
    add(ArgSpec("--rope-scaling", Kind.CHOICE, C, "RoPE 缩放方式", choices=("none", "linear", "yarn"),
                section="RoPE / YaRN", advanced=True, env="LLAMA_ARG_ROPE_SCALING_TYPE"))
    add(ArgSpec("--rope-scale", Kind.FLOAT, C, "RoPE 缩放系数", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_ROPE_SCALE"))
    add(ArgSpec("--rope-freq-base", Kind.FLOAT, C, "RoPE 基频", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_ROPE_FREQ_BASE"))
    add(ArgSpec("--rope-freq-scale", Kind.FLOAT, C, "RoPE 频率缩放", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_ROPE_FREQ_SCALE"))
    add(ArgSpec("--yarn-orig-ctx", Kind.INT, C, "YaRN 原始上下文", minimum=0, section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_YARN_ORIG_CTX"))
    add(ArgSpec("--yarn-ext-factor", Kind.FLOAT, C, "YaRN 外推系数", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_YARN_EXT_FACTOR"))
    add(ArgSpec("--yarn-attn-factor", Kind.FLOAT, C, "YaRN 注意力系数", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_YARN_ATTN_FACTOR"))
    add(ArgSpec("--yarn-beta-slow", Kind.FLOAT, C, "YaRN beta slow", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_YARN_BETA_SLOW"))
    add(ArgSpec("--yarn-beta-fast", Kind.FLOAT, C, "YaRN beta fast", section="RoPE / YaRN",
                advanced=True, env="LLAMA_ARG_YARN_BETA_FAST"))

    # ================================================================ hardware
    add(ArgSpec("--gpu-layers", Kind.STR, H, "GPU 层数 -ngl", default="auto",
                section="GPU 卸载", common=True, env="LLAMA_ARG_N_GPU_LAYERS",
                help="auto / all / 具体数字。留空则允许 --fit 自行决定。"))
    add(ArgSpec("--flash-attn", Kind.TRISTATE, H, "Flash Attention", choices=("on", "off", "auto"),
                section="GPU 卸载", common=True, env="LLAMA_ARG_FLASH_ATTN",
                help="auto 让 llama.cpp 自行判断。开启可显著降低注意力显存并提速。"))
    add(ArgSpec("--device", Kind.CSV, H, "使用设备 -dev", section="GPU 卸载", common=True,
                env="LLAMA_ARG_DEVICE", help="例如 CUDA0；填 none 走纯 CPU。"))
    add(ArgSpec("--split-mode", Kind.CHOICE, H, "多卡拆分方式", choices=("none", "layer", "row", "tensor"),
                section="多卡", advanced=True, env="LLAMA_ARG_SPLIT_MODE"))
    add(ArgSpec("--tensor-split", Kind.CSV, H, "张量分配比例 -ts", section="多卡", advanced=True,
                env="LLAMA_ARG_TENSOR_SPLIT", help="形如 3,1"))
    add(ArgSpec("--main-gpu", Kind.INT, H, "主 GPU 索引", minimum=0, default=0, section="多卡",
                advanced=True, env="LLAMA_ARG_MAIN_GPU"))
    add(ArgSpec("--override-tensor", Kind.CSV, H, "张量缓冲类型覆盖 -ot", section="多卡",
                advanced=True, env="LLAMA_ARG_OVERRIDE_TENSOR",
                help="形如 blk\\.(\\d+)\\.ffn_.*=CPU"))
    add(ArgSpec("--cpu-moe", Kind.BOOL, H, "MoE 权重全部放 CPU", default=False, section="MoE",
                env="LLAMA_ARG_CPU_MOE",
                help="显存不够放下 MoE 模型时最有效的手段。"))
    add(ArgSpec("--n-cpu-moe", Kind.INT, H, "前 N 层 MoE 放 CPU", minimum=0, section="MoE",
                env="LLAMA_ARG_N_CPU_MOE"))
    add(ArgSpec("--n-cpu-ffn", Kind.INT, H, "前 N 层 FFN 放 CPU（稠密）", minimum=0, section="MoE",
                advanced=True, env="LLAMA_ARG_N_CPU_FFN"))
    add(ArgSpec("--fit", Kind.TRISTATE, H, "自动适配显存 --fit", choices=("on", "off"),
                default="on", section="显存适配", env="LLAMA_ARG_FIT",
                help="自动调整你未显式设置的参数以塞进显存。本 GUI 只发送你手动填写的参数，因此它默认有效。"))
    add(ArgSpec("--fit-target", Kind.CSV, H, "适配预留 margin (MiB)", section="显存适配",
                advanced=True, env="LLAMA_ARG_FIT_TARGET"))
    add(ArgSpec("--fit-ctx", Kind.INT, H, "适配最小上下文", minimum=0, section="显存适配",
                advanced=True, env="LLAMA_ARG_FIT_CTX"))
    add(ArgSpec("--cache-type-k", Kind.CHOICE, H, "KV cache K 类型", choices=tuple(DTYPE_CHOICES),
                default=DEFAULT_CACHE_TYPE_K, section="KV cache", common=True,
                env="LLAMA_ARG_CACHE_TYPE_K"))
    add(ArgSpec("--cache-type-v", Kind.CHOICE, H, "KV cache V 类型", choices=tuple(DTYPE_CHOICES),
                default=DEFAULT_CACHE_TYPE_V, section="KV cache", common=True,
                env="LLAMA_ARG_CACHE_TYPE_V"))
    add(ArgSpec("--kv-offload", Kind.BOOL, H, "KV cache 放显存", default=True,
                neg="--no-kv-offload", section="KV cache", advanced=True, env="LLAMA_ARG_KV_OFFLOAD"))
    add(ArgSpec("--cache-ram", Kind.INT, H, "提示词缓存上限 (MiB)", minimum=-1, default=8192,
                section="KV cache", advanced=True, env="LLAMA_ARG_CACHE_RAM",
                help="-1 不限，0 关闭。默认 8192 MiB 会常驻内存，可适当调小。"))
    add(ArgSpec("--cache-idle-slots", Kind.BOOL, H, "空闲槽写入提示词缓存", default=True,
                neg="--no-cache-idle-slots", section="KV cache", advanced=True,
                env="LLAMA_ARG_CACHE_IDLE_SLOTS"))
    add(ArgSpec("--load-mode", Kind.CHOICE, H, "模型加载模式", choices=tuple(LOAD_MODES),
                default="auto", section="加载", advanced=True, env="LLAMA_ARG_LOAD_MODE"))
    add(ArgSpec("--lazy-mode", Kind.CHOICE, H, "惰性读取模式", choices=("on", "auto", "off"),
                default="auto", section="加载", advanced=True, env="LLAMA_ARG_LAZY_MODE",
                help="按需从磁盘读大张量（需要 mmap），可显著降低常驻内存。"))
    # --no-host is itself a negation, so it is modelled as an ordinary boolean
    # whose *flag* is the negative form: unticked means "not set", which leaves
    # llama.cpp allocating a host buffer as usual, and ticking it emits the flag.
    # Modelling it the other way round (ticking = "use host buffer") would mean
    # an untouched checkbox silently disabled the buffer.
    add(ArgSpec("--no-host", Kind.BOOL, H, "绕过 host 缓冲（--no-host）", default=False,
                section="加载", advanced=True, env="LLAMA_ARG_NO_HOST",
                help="勾选后禁用 host 缓冲，用于腾出额外缓冲；默认不勾选即保持默认行为。"))
    add(ArgSpec("--repack", Kind.BOOL, H, "权重重打包", default=True, neg="--no-repack",
                section="加载", advanced=True, env="LLAMA_ARG_REPACK"))
    add(ArgSpec("--op-offload", Kind.BOOL, H, "host 算子卸载到设备", default=True,
                neg="--no-op-offload", section="加载", advanced=True))
    add(ArgSpec("--rpc", Kind.CSV, H, "RPC 后端服务器", section="RPC / 分布式", advanced=True,
                env="LLAMA_ARG_RPC", help="形如 host:port,host:port，需要配合 ggml-rpc-server。"))
    add(ArgSpec("--check-tensors", Kind.BOOL, H, "校验张量数值", default=False, section="加载",
                advanced=True, help="加载极慢，仅排障用。"))
    add(ArgSpec("--override-kv", Kind.CSV, H, "覆盖模型元数据", section="高级", advanced=True,
                help="形如 tokenizer.ggml.add_bos_token=bool:false"))

    # ================================================================= sampling
    add(ArgSpec("--temp", Kind.FLOAT, S, "温度 temperature", default=0.8, minimum=0.0,
                section="基础采样", common=True))
    add(ArgSpec("--top-k", Kind.INT, S, "top-k", default=40, minimum=0, section="基础采样",
                common=True, env="LLAMA_ARG_TOP_K", help="0 = 关闭。"))
    add(ArgSpec("--top-p", Kind.FLOAT, S, "top-p", default=0.95, minimum=0.0, maximum=1.0,
                section="基础采样", common=True, help="1.0 = 关闭。"))
    add(ArgSpec("--min-p", Kind.FLOAT, S, "min-p", default=0.05, minimum=0.0, maximum=1.0,
                section="基础采样", common=True, help="0.0 = 关闭。"))
    add(ArgSpec("--top-nsigma", Kind.FLOAT, S, "top-n-sigma", default=-1.0, section="基础采样",
                advanced=True))
    add(ArgSpec("--xtc-probability", Kind.FLOAT, S, "XTC 概率", default=0.0, minimum=0.0,
                maximum=1.0, section="基础采样", advanced=True))
    add(ArgSpec("--xtc-threshold", Kind.FLOAT, S, "XTC 阈值", default=0.1, section="基础采样",
                advanced=True))
    add(ArgSpec("--typical", Kind.FLOAT, S, "locally typical", default=1.0, minimum=0.0,
                maximum=1.0, section="基础采样", advanced=True))
    add(ArgSpec("--adaptive-target", Kind.FLOAT, S, "adaptive-p 目标", default=-1.0,
                section="基础采样", advanced=True,
                help="0.0~1.0，负数关闭。"))
    add(ArgSpec("--adaptive-decay", Kind.FLOAT, S, "adaptive-p 衰减", default=0.9, minimum=0.0,
                maximum=0.99, section="基础采样", advanced=True))
    add(ArgSpec("--seed", Kind.INT, S, "随机种子", default=-1, section="基础采样",
                help="-1 = 每次随机。"))
    add(ArgSpec("--samplers", Kind.STR, S, "采样器顺序", section="基础采样", advanced=True,
                help="分号分隔。留空使用内置顺序。"))
    add(ArgSpec("--sampler-seq", Kind.STR, S, "采样器简写序列", section="基础采样", advanced=True))
    add(ArgSpec("--ignore-eos", Kind.BOOL, S, "忽略 EOS 继续生成", default=False,
                section="基础采样", advanced=True))
    add(ArgSpec("--repeat-last-n", Kind.INT, S, "重复惩罚回看窗口", default=64, minimum=-1,
                section="重复惩罚"))
    add(ArgSpec("--repeat-penalty", Kind.FLOAT, S, "重复惩罚", default=1.0, minimum=0.0,
                section="重复惩罚", common=True, help="1.0 = 关闭；>1 抑制重复。"))
    add(ArgSpec("--presence-penalty", Kind.FLOAT, S, "存在惩罚", default=0.0, section="重复惩罚",
                common=True))
    add(ArgSpec("--frequency-penalty", Kind.FLOAT, S, "频率惩罚", default=0.0, section="重复惩罚",
                common=True))
    add(ArgSpec("--dry-multiplier", Kind.FLOAT, S, "DRY 强度", default=0.0, minimum=0.0,
                section="DRY", help="0.0 = 关闭。对抑制复读非常有效。"))
    add(ArgSpec("--dry-base", Kind.FLOAT, S, "DRY base", default=1.75, section="DRY"))
    add(ArgSpec("--dry-allowed-length", Kind.INT, S, "DRY 允许长度", default=2, section="DRY"))
    add(ArgSpec("--dry-penalty-last-n", Kind.INT, S, "DRY 回看窗口", default=64, minimum=-1,
                section="DRY"))
    add(ArgSpec("--dry-sequence-breaker", Kind.REPEAT, S, "DRY 断句符", section="DRY",
                advanced=True, help='填 none 表示不使用任何断句符。'))
    add(ArgSpec("--dynatemp-range", Kind.FLOAT, S, "动态温度范围", default=0.0, minimum=0.0,
                section="动态温度", advanced=True))
    add(ArgSpec("--dynatemp-exp", Kind.FLOAT, S, "动态温度指数", default=1.0, section="动态温度",
                advanced=True))
    add(ArgSpec("--mirostat", Kind.INT, S, "Mirostat 模式", default=0, minimum=0, maximum=2,
                section="Mirostat", help="0 关闭 / 1 Mirostat / 2 Mirostat 2.0"))
    add(ArgSpec("--mirostat-lr", Kind.FLOAT, S, "Mirostat 学习率", default=0.1, section="Mirostat"))
    add(ArgSpec("--mirostat-ent", Kind.FLOAT, S, "Mirostat 目标熵", default=5.0, section="Mirostat"))
    add(ArgSpec("--logit-bias", Kind.REPEAT, S, "logit 偏置", section="约束生成", advanced=True,
                help="形如 15043+1 / 15043-1"))
    add(ArgSpec("--grammar", Kind.TEXT, S, "语法约束 (BNF)", section="约束生成"))
    add(ArgSpec("--grammar-file", Kind.PATH, S, "语法文件", section="约束生成"))
    add(ArgSpec("--json-schema", Kind.TEXT, S, "JSON Schema", section="约束生成",
                help='例如 {"type": "object"}'))
    add(ArgSpec("--json-schema-file", Kind.PATH, S, "JSON Schema 文件", section="约束生成"))
    add(ArgSpec("--backend-sampling", Kind.BOOL, S, "后端采样（实验）", default=False,
                section="约束生成", experimental=True, env="LLAMA_ARG_BACKEND_SAMPLING"))

    # ============================================================== speculative
    add(ArgSpec("--spec-type", Kind.CHOICE, P,
                "投机解码类型",
                choices=("none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
                         "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                         "ngram-mod", "ngram-cache"),
                default="none", section="总开关", env="LLAMA_ARG_SPEC_TYPE",
                help="ngram 系不需要草稿模型，适合代码补全与重复性文本。"))
    add(ArgSpec("--spec-default", Kind.BOOL, P, "使用默认投机配置", default=False, section="总开关"))
    add(ArgSpec("--spec-draft-model", Kind.PATH, P, "草稿模型", section="草稿模型",
                env="LLAMA_ARG_SPEC_DRAFT_MODEL"))
    add(ArgSpec("--spec-draft-hf", Kind.STR, P, "草稿模型 HF 仓库", section="草稿模型",
                env="LLAMA_ARG_SPEC_DRAFT_HF_REPO"))
    add(ArgSpec("--spec-draft-ngl", Kind.STR, P, "草稿模型 GPU 层数", section="草稿模型"))
    add(ArgSpec("--spec-draft-device", Kind.CSV, P, "草稿模型设备", section="草稿模型",
                advanced=True))
    add(ArgSpec("--spec-draft-threads", Kind.INT, P, "草稿模型线程", minimum=-1,
                section="草稿模型", advanced=True))
    add(ArgSpec("--spec-draft-threads-batch", Kind.INT, P, "草稿模型批处理线程", minimum=-1,
                section="草稿模型", advanced=True))
    add(ArgSpec("--spec-draft-type-k", Kind.CHOICE, P, "草稿 KV K 类型",
                choices=tuple(DTYPE_CHOICES), default=DEFAULT_CACHE_TYPE_K, section="草稿模型",
                advanced=True, env="LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"))
    add(ArgSpec("--spec-draft-type-v", Kind.CHOICE, P, "草稿 KV V 类型",
                choices=tuple(DTYPE_CHOICES), default=DEFAULT_CACHE_TYPE_V, section="草稿模型",
                advanced=True, env="LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"))
    add(ArgSpec("--spec-draft-cpu-moe", Kind.BOOL, P, "草稿 MoE 放 CPU", default=False,
                section="草稿模型", advanced=True, env="LLAMA_ARG_SPEC_DRAFT_CPU_MOE"))
    add(ArgSpec("--spec-draft-n-max", Kind.INT, P, "最大草稿 token 数", default=3, minimum=0,
                section="草稿模型", env="LLAMA_ARG_SPEC_DRAFT_N_MAX"))
    add(ArgSpec("--spec-draft-n-min", Kind.INT, P, "最小草稿 token 数", minimum=0,
                section="草稿模型"))
    add(ArgSpec("--spec-draft-p-split", Kind.FLOAT, P, "草稿分裂概率 p-split", minimum=0.0,
                maximum=1.0, section="草稿模型", advanced=True))
    add(ArgSpec("--spec-draft-p-min", Kind.FLOAT, P, "草稿最小概率 p-min", default=0.0, minimum=0.0,
                maximum=1.0, section="草稿模型", advanced=True))
    add(ArgSpec("--spec-synth-len", Kind.FLOAT, P, "合成接受长度目标", section="合成", advanced=True))
    add(ArgSpec("--spec-synth-rates", Kind.CSV, P, "合成接受率序列", section="合成", advanced=True))
    add(ArgSpec("--spec-ngram-mod-n-min", Kind.INT, P, "ngram-mod 最小 N", minimum=0, section="ngram"))
    add(ArgSpec("--spec-ngram-mod-n-max", Kind.INT, P, "ngram-mod 最大 N", minimum=0, section="ngram"))
    add(ArgSpec("--spec-ngram-mod-n-match", Kind.INT, P, "ngram-mod 匹配长度", minimum=0,
                section="ngram"))
    add(ArgSpec("--spec-ngram-simple-size-n", Kind.INT, P, "ngram-simple N", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-simple-size-m", Kind.INT, P, "ngram-simple M", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-simple-min-hits", Kind.INT, P, "ngram-simple 最小命中", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-map-k-size-n", Kind.INT, P, "ngram-map-k N", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-map-k-size-m", Kind.INT, P, "ngram-map-k M", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-map-k-min-hits", Kind.INT, P, "ngram-map-k 最小命中", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-map-k4v-size-n", Kind.INT, P, "ngram-map-k4v N", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-map-k4v-size-m", Kind.INT, P, "ngram-map-k4v M", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--spec-ngram-map-k4v-min-hits", Kind.INT, P, "ngram-map-k4v 最小命中", minimum=0,
                section="ngram", advanced=True))
    add(ArgSpec("--lookup-cache-static", Kind.PATH, P, "静态查找缓存", section="ngram",
                advanced=True))
    add(ArgSpec("--lookup-cache-dynamic", Kind.PATH, P, "动态查找缓存", section="ngram",
                advanced=True))

    # =================================================================== server
    add(ArgSpec("--host", Kind.STR, V, "监听地址", default="127.0.0.1", section="监听",
                common=True, env="LLAMA_ARG_HOST",
                help="默认仅本机可访问。改成 0.0.0.0 会暴露到局域网。"))
    # No hard-coded default: llama.cpp changed it (8080 -> 9931 around build
    # 12000).  Left empty, the GUI reads the installed build's own default.
    add(ArgSpec("--port", Kind.INT, V, "端口", minimum=1, maximum=65535,
                section="监听", common=True, env="LLAMA_ARG_PORT",
                help="留空使用本 build 的默认端口（以 --help 为准，新版为 9931）。"
                     "GUI 始终以服务日志里实际的 listening on 地址为准。"))
    add(ArgSpec("--reuse-port", Kind.BOOL, V, "允许端口复用", default=False, section="监听",
                advanced=True, env="LLAMA_ARG_REUSE_PORT"))
    add(ArgSpec("--api-key", Kind.REPEAT, V, "API 密钥", section="认证", common=True,
                secret=True,
                help="设置后客户端需带 Authorization: Bearer <key>。"))
    add(ArgSpec("--api-key-file", Kind.PATH, V, "API 密钥文件", section="认证"))
    # Note: --ssl-key-file is a *path*, not key material, so it is not secret.
    add(ArgSpec("--ssl-key-file", Kind.PATH, V, "SSL 私钥 (PEM)", section="认证", advanced=True))
    add(ArgSpec("--ssl-cert-file", Kind.PATH, V, "SSL 证书 (PEM)", section="认证", advanced=True))
    add(ArgSpec("--api-prefix", Kind.STR, V, "API 路径前缀", section="监听", advanced=True,
                env="LLAMA_ARG_API_PREFIX", help="不带结尾斜杠，例如 /llama。"))
    add(ArgSpec("--path", Kind.PATH, V, "静态文件目录", section="监听", advanced=True,
                env="LLAMA_ARG_STATIC_PATH"))
    add(ArgSpec("--cors-origins", Kind.STR, V, "CORS 允许来源", default="*", section="CORS",
                advanced=True, env="LLAMA_ARG_CORS_ORIGINS"))
    add(ArgSpec("--cors-methods", Kind.STR, V, "CORS 允许方法", section="CORS", advanced=True,
                env="LLAMA_ARG_CORS_METHODS"))
    add(ArgSpec("--cors-headers", Kind.STR, V, "CORS 允许头", section="CORS", advanced=True,
                env="LLAMA_ARG_CORS_HEADERS"))
    add(ArgSpec("--cors-credentials", Kind.BOOL, V, "CORS 允许凭证", default=True,
                neg="--no-cors-credentials", section="CORS", advanced=True,
                env="LLAMA_ARG_CORS_CREDENTIALS"))
    add(ArgSpec("--timeout", Kind.INT, V, "读写超时 (秒)", default=3600, minimum=0, section="行为",
                advanced=True))
    add(ArgSpec("--sse-ping-interval", Kind.INT, V, "SSE 心跳间隔 (秒)", default=30,
                section="行为", advanced=True, help="-1 = 关闭"))
    add(ArgSpec("--threads-http", Kind.INT, V, "HTTP 线程数", minimum=-1, section="行为",
                advanced=True))
    add(ArgSpec("--cache-prompt", Kind.BOOL, V, "提示词缓存", default=True,
                neg="--no-cache-prompt", section="行为", env="LLAMA_ARG_CACHE_PROMPT"))
    add(ArgSpec("--cache-reuse", Kind.INT, V, "KV 复用最小块", minimum=0, section="行为",
                advanced=True))
    add(ArgSpec("--slot-prompt-similarity", Kind.FLOAT, V, "槽位相似度阈值", section="行为",
                advanced=True))
    add(ArgSpec("--slot-save-path", Kind.PATH, V, "槽位缓存保存目录", section="行为",
                advanced=True))
    add(ArgSpec("--warmup", Kind.BOOL, V, "启动预热", default=True, neg="--no-warmup",
                section="行为", advanced=True))
    add(ArgSpec("--sleep-idle-seconds", Kind.INT, V, "空闲休眠秒数", minimum=-1, section="行为",
                advanced=True, help="-1 关闭。休眠后释放显存，下次请求再唤醒。"))
    add(ArgSpec("--metrics", Kind.BOOL, V, "启用 Prometheus 指标", default=False, section="端点",
                env="LLAMA_ARG_ENDPOINT_METRICS"))
    add(ArgSpec("--props", Kind.BOOL, V, "允许 POST /props 改属性", default=False, section="端点",
                env="LLAMA_ARG_ENDPOINT_PROPS"))
    add(ArgSpec("--slots", Kind.BOOL, V, "暴露 /slots 监控端点", default=True, neg="--no-slots",
                section="端点", env="LLAMA_ARG_ENDPOINT_SLOTS"))
    add(ArgSpec("--embedding", Kind.BOOL, V, "仅嵌入模式", default=False, section="端点",
                help="需要专门的 embedding 模型（pooling 不是 none）。"))
    add(ArgSpec("--reranking", Kind.BOOL, V, "启用重排序端点", default=False, section="端点"))
    add(ArgSpec("--pooling", Kind.CHOICE, V, "嵌入池化方式",
                choices=("none", "mean", "cls", "last", "rank"), section="端点", advanced=True,
                env="LLAMA_ARG_POOLING"))
    add(ArgSpec("--embd-normalize", Kind.INT, V, "嵌入归一化", default=2, section="端点",
                advanced=True))
    add(ArgSpec("--ui", Kind.BOOL, V, "启用内置 Web UI", default=True, neg="--no-webui",
                section="Web UI", common=True,
                help="llama-server 自带的网页界面。一键启动指的就是打开它。"))
    add(ArgSpec("--ui-config-file", Kind.PATH, V, "Web UI 默认配置", section="Web UI",
                advanced=True, env="LLAMA_ARG_UI_CONFIG_FILE"))
    add(ArgSpec("--ui-mcp-proxy", Kind.BOOL, V, "Web UI MCP 代理", default=False,
                neg="--no-ui-mcp-proxy", section="安全（谨慎）", danger=True,
                env="LLAMA_ARG_UI_MCP_PROXY",
                help="实验性。不要在不可信环境开启，它会打开 CORS 代理。"))
    add(ArgSpec("--tools", Kind.STR, V, "内置工具集", section="安全（谨慎）", danger=True,
                env="LLAMA_ARG_TOOLS",
                help="实验性：read_file / write_file / exec_shell_command 等。填 all 启用全部。"))
    add(ArgSpec("--tools-runtime", Kind.STR, V, "工具运行环境", section="安全（谨慎）", danger=True,
                env="LLAMA_ARG_TOOLS_RUNTIME",
                help="例如 docker:<image> 或 ssh:<target>，强烈建议隔离运行。"))
    add(ArgSpec("--mcp-servers-config", Kind.PATH, V, "MCP 服务器配置", section="安全（谨慎）",
                danger=True))
    add(ArgSpec("--mcp-servers-json", Kind.TEXT, V, "MCP 服务器 JSON", section="安全（谨慎）",
                danger=True))
    add(ArgSpec("--agent", Kind.BOOL, V, "启用 agent 模式", default=False, neg="--no-agent",
                section="安全（谨慎）", danger=True,
                help="等价于打开 CORS 代理与全部内置工具。"))

    # ================================================================== logging
    add(ArgSpec("--verbosity", Kind.INT, L, "日志详细度", minimum=0, maximum=5, default=3,
                section="日志", env="LLAMA_ARG_LOG_VERBOSITY",
                help="0 通用 / 1 错误 / 2 警告 / 3 信息 / 4 追踪 / 5 调试"))
    add(ArgSpec("--verbose", Kind.BOOL, L, "最高详细度 (-v)", default=False, section="日志",
                help="打开后日志量极大，排障时用。"))
    add(ArgSpec("--log-file", Kind.PATH, L, "额外写入日志文件", section="日志",
                env="LLAMA_ARG_LOG_FILE"))
    add(ArgSpec("--log-jsonl", Kind.BOOL, L, "JSONL 格式日志", default=False, section="日志",
                advanced=True, env="LLAMA_ARG_LOG_JSONL",
                help="本 GUI 会解析该格式并按级别着色。"))
    add(ArgSpec("--log-colors", Kind.TRISTATE, L, "彩色日志", choices=("on", "off", "auto"),
                default="off", section="日志", env="LLAMA_ARG_LOG_COLORS",
                help="GUI 管道输出下固定建议 off，避免 ANSI 转义序列。"))
    add(ArgSpec("--log-timestamps", Kind.BOOL, L, "日志时间戳", default=False,
                neg="--no-log-timestamps", section="日志", advanced=True,
                env="LLAMA_ARG_LOG_TIMESTAMPS"))
    add(ArgSpec("--log-prefix", Kind.BOOL, L, "日志前缀", default=False, neg="--no-log-prefix",
                section="日志", advanced=True, env="LLAMA_ARG_LOG_PREFIX"))
    add(ArgSpec("--log-prompts-dir", Kind.PATH, L, "记录请求 prompt 目录", section="日志",
                advanced=True, help="会把所有 prompt 落盘，注意隐私。"))
    add(ArgSpec("--perf", Kind.BOOL, L, "输出内部性能计时", default=False, section="日志",
                help="日志中出现 llama_perf_* 计时行，便于抽取 t/s。"))

    return s


DYNAMIC_CHOICES = ("on", "off", "auto")

#: KV cache element types accepted by ``-ctk`` / ``-ctv`` in this build.
DTYPE_CHOICES = (
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
)

#: Values for ``--load-mode``.
LOAD_MODES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")

#: The full table, keyed by canonical long flag.
SPECS: Dict[str, ArgSpec] = {spec.flag: spec for spec in _specs()}

#: Specs in declaration order, which is also the on-screen order within a
#: section.
SPEC_ORDER: Tuple[str, ...] = tuple(SPECS)

#: Environment variables the server reads; shown in the API/环境 page so users
#: can see which ones a launch will effectively override.
KNOWN_ENV: Tuple[str, ...] = tuple(
    sorted({spec.env for spec in SPECS.values() if spec.env})
)


def specs_for(group: Group) -> List[ArgSpec]:
    """All specs belonging to *group*, in declaration order."""
    return [SPECS[f] for f in SPEC_ORDER if SPECS[f].group is group]


def sections_for(group: Group) -> List[str]:
    """Section names within *group*, in first-seen order."""
    seen: List[str] = []
    for spec in specs_for(group):
        if spec.section and spec.section not in seen:
            seen.append(spec.section)
    return seen


def get(flag: str) -> Optional[ArgSpec]:
    """Look up a spec by canonical flag."""
    return SPECS.get(flag)


def canonical(flag: str) -> str:
    """Map any documented long form to this table's canonical entry.

    Returns *flag* unchanged when it is not in the table, so callers can still
    round-trip unknown arguments.
    """
    if flag in SPECS:
        return flag
    for spec in SPECS.values():
        if flag == spec.neg:
            return spec.flag
    return flag

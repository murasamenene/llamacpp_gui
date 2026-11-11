"""Turn a :class:`~llamagui.config.Config` into a llama-server command line.

Two properties matter more than anything else here:

1. **Only explicitly-set arguments are emitted.**  llama.cpp decides an argument
   is "set by the user" from the command line, and ``--fit`` only adjusts the
   arguments that were *not* set.  Emitting a GUI default would silently
   disable that behaviour, so a value that the user never touched must never
   appear in argv.
2. **Validation is layered and honest.**  Hard errors block the launch; soft
   warnings are surfaced but let the user proceed, because llama.cpp is the
   final authority and occasionally rejects a configuration the estimator
   thought was fine.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .argspec import SPECS, get as get_spec
from .config import Config
from .estimate import DEFAULT_CACHE_TYPE_K, DEFAULT_CACHE_TYPE_V, KV_DTYPE_BYTES

#: Arguments that the server derives from others, or that the GUI manages
#: itself, and which must therefore never be forwarded verbatim.
MANAGED_FLAGS = frozenset({"--help", "--usage", "--version", "--list-devices",
                           "--completion-bash", "--cache-list"})

#: Flags accepted by the launcher that are not in the curated table.  Emitted
#: verbatim when present in the config, which keeps preset files forward
#: compatible with a newer llama.cpp build.
_PASSTHROUGH_PREFIX = "--"


@dataclass
class Validation:
    """Collected problems for one candidate command line."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def extend(self, other: "Validation") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.infos.extend(other.infos)


@dataclass
class BuiltCommand:
    """Result of :func:`build`."""

    argv: List[str]
    validation: Validation
    env: Dict[str, str] = field(default_factory=dict)
    #: Flags that were present in the config but unknown to this build.
    unknown: List[str] = field(default_factory=list)
    #: Whether the launcher is in multi-model router mode (no ``-m``).
    router_mode: bool = False

    @property
    def ok(self) -> bool:
        return self.validation.ok

    def preview(self, executable: str = "llama-server.exe") -> str:
        """A copy-pasteable one-liner."""
        return format_command([executable, *self.argv])

    def multiline(self, executable: str = "llama-server.exe", width: int = 100) -> str:
        """Wrapped form for the console and log header."""
        tokens = [executable, *self.argv]
        lines: List[str] = []
        current = ""
        for token in tokens:
            piece = quote_token(token)
            if current and len(current) + 1 + len(piece) > width:
                lines.append(current)
                current = "    " + piece
            else:
                current = f"{current} {piece}".strip() if current else piece
        if current:
            lines.append(current)
        return " \\\n".join(lines)


# -------------------------------------------------------------------- quoting


def quote_token(token: str) -> str:
    """Quote *token* for display only; ``argv`` passing never needs quoting."""
    if token == "" or any(ch in token for ch in ' \t"\\$&|<>^'):
        escaped = token.replace('"', '\\"')
        return f'"{escaped}"'
    return token


def format_command(tokens: Sequence[str]) -> str:
    return " ".join(quote_token(t) for t in tokens)


# --------------------------------------------------------------------- build


def build(
    config: Config,
    *,
    help_doc: Any = None,
    include_router_flags: bool = True,
    base_dir: Optional[Path] = None,
) -> BuiltCommand:
    """Build the ``llama-server`` argument list from *config*.

    *help_doc* is an optional :class:`~llamagui.helpparse.HelpDoc`.  When given,
    any config flag this build does not document is reported in
    :attr:`BuiltCommand.unknown` and dropped rather than emitted, because
    llama-server aborts on an unknown argument.

    *base_dir* is the working directory the server will be started in, used to
    resolve relative file paths during validation.  It defaults to the current
    directory.
    """
    validation = Validation()
    argv: List[str] = []
    unknown: List[str] = []
    workdir = Path(base_dir) if base_dir is not None else Path.cwd()

    values = config.values

    # Router mode: the server switches to multi-model routing when no -m/-hf is
    # given and --models-dir/--models-preset is provided.
    has_model = _nonempty(values.get("--model")) or _nonempty(values.get("--hf-repo"))
    has_router = _nonempty(values.get("--models-dir")) or _nonempty(values.get("--models-preset"))
    router_mode = has_router and not has_model
    if has_router and has_model:
        validation.error(
            "路由模式（--models-dir/--models-preset）与 --model/--hf-repo 不能同时使用："
            "llama-server 只在不指定模型时才进入路由模式。"
        )
    if not has_model and not has_router:
        validation.error("未指定模型：请填写本地 GGUF 路径，或 HuggingFace 仓库，或启用路由模式。")

    for flag in _ordered_flags(values):
        if flag in MANAGED_FLAGS:
            continue
        raw = values.get(flag)
        if raw is None:
            continue

        spec = get_spec(flag)
        if spec is None:
            if help_doc is not None:
                opt = help_doc.get(flag)
                if opt is None or opt.removed:
                    # Either this build does not document the flag at all, or it
                    # documents it as removed.  Never forward: llama-server
                    # aborts on an unknown or removed argument, and the user
                    # would have no idea which preset entry caused it.
                    unknown.append(flag)
                    continue
            # Documented but not curated: forward so presets written for a newer
            # build keep working.
            tokens = _passthrough_tokens(flag, raw)
            if tokens:
                argv.extend(tokens)
            continue

        if help_doc is not None:
            opt = help_doc.get(flag)
            if opt is None or opt.removed:
                unknown.append(flag)
                continue

        try:
            coerced = spec.coerce(raw)
        except ValueError as exc:
            validation.error(str(exc))
            continue
        if coerced is None:
            continue
        argv.extend(spec.serialize(coerced))

    validate(config, validation, router_mode=router_mode, base_dir=workdir)

    env = {str(k): str(v) for k, v in config.env.items() if str(k).strip()}
    return BuiltCommand(
        argv=argv,
        validation=validation,
        env=env,
        unknown=unknown,
        router_mode=router_mode,
    )


def _ordered_flags(values: Dict[str, Any]) -> List[str]:
    """Config flags in table order, with unknown flags last but stable."""
    known = [f for f in SPECS if f in values]
    extra = [f for f in values if f not in SPECS]
    extra.sort()
    return known + extra


#: Flags that carry meaning even when set to the value the table calls the
#: default, because leaving them out would change llama.cpp's behaviour.  The
#: GUI records a key only when the user edits the field, so presence in the
#: config already means "chosen deliberately"; this set exists to document
#: which of those options are *sensitive* to being written explicitly, and is
#: used by the UI to keep the field visibly set rather than showing it blank.
ALWAYS_EMIT: frozenset = frozenset({
    "--fit",              # --fit on/off/auto defaults to on; omitting re-enables fit
    "--flash-attn",       # tri-state auto/on/off
    "--reasoning",        # tri-state auto/on/off
    "--load-mode",        # auto is a meaningful mode, not "unset"
    "--lazy-mode",        # auto is a meaningful mode
    "--log-colors",       # "off" must be explicit to strip ANSI in a pipe
    "--cache-type-k",
    "--cache-type-v",
})


def _passthrough_tokens(flag: str, raw: Any) -> List[str]:
    """Serialise an uncurated flag without knowing its type."""
    if flag.startswith("-"):
        if raw is True:
            return [flag]
        if raw is False:
            return []
        if isinstance(raw, (list, tuple)):
            out: List[str] = []
            for item in raw:
                out.extend([flag, str(item)])
            return out
        if raw is None or raw == "":
            return []
        return [flag, str(raw)]
    return []


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


# ---------------------------------------------------------------- validation


def validate(
    config: Config,
    result: Validation,
    *,
    router_mode: bool = False,
    base_dir: Optional[Path] = None,
) -> None:
    """Cross-argument checks.  Errors block; warnings inform."""
    v = config.values
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    def resolve(value: object) -> Path:
        """Resolve a user-supplied path the way the child process will see it."""
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (base / path)

    ctx = _as_int(v.get("--ctx-size"))
    batch = _as_int(v.get("--batch-size"))
    ubatch = _as_int(v.get("--ubatch-size"))
    parallel = _as_int(v.get("--parallel"))
    kv_unified = v.get("--kv-unified")
    ctk = str(v.get("--cache-type-k") or DEFAULT_CACHE_TYPE_K)
    ctv = str(v.get("--cache-type-v") or DEFAULT_CACHE_TYPE_V)

    if ubatch is not None and ubatch > 0:
        if ubatch % 32 != 0:
            result.warn(f"物理批 -ub={ubatch} 通常应为 32 的倍数，否则可能被 llama.cpp 调整。")
        if batch is not None and batch > 0 and ubatch > batch:
            result.error(f"物理批 -ub={ubatch} 不能大于逻辑批 -b={batch}。")
        if ctx is not None and ctx > 0 and ubatch > ctx:
            result.error(f"物理批 -ub={ubatch} 不能大于上下文 -c={ctx}。")
    if batch is not None and ctx is not None and ctx > 0 and batch > ctx:
        result.warn(f"逻辑批 -b={batch} 大于上下文 -c={ctx}，通常没有意义。")

    if parallel is not None and parallel > 1:
        if kv_unified is False:
            result.warn(
                f"已关闭统一 KV 缓冲且 -np={parallel}：每个槽位会各占 -c 的上下文，"
                f"KV cache 总量约为单槽的 {parallel} 倍。"
            )
        elif ctx is not None and ctx > 0:
            result.info(
                f"-np={parallel} 共享统一 KV 池，总量仍为 -c={ctx}；"
                f"单槽可用上下文约为 {max(1, ctx // parallel)}。"
            )
        if result.ok and parallel > 1:
            result.info("客户端需要能处理多槽并发；单客户端使用时 -np=1 更省显存。")

    if ctk in KV_DTYPE_BYTES and ctv in KV_DTYPE_BYTES:
        if ctv != "f16" and ctv not in ("f32", "bf16"):
            result.info(f"V cache 使用 {ctv}，部分后端对量化 V 的支持有限，速度可能下降。")
    elif ctk or ctv:
        result.warn(f"KV cache 类型 {ctk}/{ctv} 在本 build 中可能不被支持。")

    host = str(v.get("--host") or "127.0.0.1").strip()
    exposed = host not in ("", "127.0.0.1", "localhost", "::1", "[::1]")
    if exposed:
        result.warn(f"--host={host} 会把服务暴露到非本机地址。")
        risky = [f for f in ("--tools", "--agent", "--ui-mcp-proxy", "--mcp-servers-config",
                             "--mcp-servers-json") if _truthy(v.get(f))]
        if risky:
            result.error(
                "拒绝启动：服务已绑定到非本机地址，同时启用了内置工具/agent（"
                + "、".join(risky)
                + "）。这会允许远端执行本机命令，请先关闭这些选项或改回 127.0.0.1。"
            )

    if v.get("--api-key") and exposed:
        result.info("已设置 API 密钥，远端访问需要 Authorization: Bearer <key>。")

    model = v.get("--model")
    if _nonempty(model):
        path = resolve(model)
        if not path.is_file():
            result.error(f"模型文件不存在：{path}")
        elif path.suffix.lower() != ".gguf":
            result.warn(f"模型文件扩展名不是 .gguf：{path.name}")

    mmproj = v.get("--mmproj")
    if _nonempty(mmproj) and not resolve(mmproj).is_file():
        result.error(f"多模态投影文件不存在：{resolve(mmproj)}")

    for flag in ("--lora", "--lora-scaled", "--control-vector", "--grammar-file",
                 "--json-schema-file", "--api-key-file", "--ssl-key-file", "--ssl-cert-file",
                 "--chat-template-file", "--lookup-cache-static", "--lookup-cache-dynamic"):
        raw = v.get(flag)
        if raw is None:
            continue
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        for item in items:
            for candidate in _file_operands(flag, item):
                if not resolve(candidate).is_file():
                    result.error(f"{flag} 指向的文件不存在：{resolve(candidate)}")

    spec_type = str(v.get("--spec-type") or "none")
    # --spec-type is a comma-separated list.  draft-mtp uses the MTP head built
    # into the main model, so it is the one draft-* mode without a draft model.
    spec_types = [t.strip() for t in spec_type.split(",") if t.strip() and t.strip() != "none"]
    if spec_types:
        needs_draft = [t for t in spec_types if t.startswith("draft") and t != "draft-mtp"]
        if needs_draft and not _nonempty(v.get("--spec-draft-model")) \
                and not _nonempty(v.get("--spec-draft-hf")):
            result.warn(f"--spec-type={','.join(needs_draft)} 需要草稿模型："
                        "请设置 --spec-draft-model 或 --spec-draft-hf。")
        if any(t.startswith("ngram") for t in spec_types) \
                and not _nonempty(v.get("--spec-ngram-mod-n-match")):
            result.info("ngram 系投机解码不需要草稿模型，适合代码补全与模板化文本。")
    else:
        spec_type = "none"

    if _truthy(v.get("--embedding")) and spec_type != "none":
        result.warn("嵌入模式通常不需要投机解码。")

    if router_mode:
        result.info("路由模式：未指定 --model，llama-server 将按需加载模型。")

    if _truthy(v.get("--agent")) or _truthy(v.get("--tools")):
        result.warn("已启用实验性内置工具/agent：这些工具可读写本机文件、执行命令。")

    if v.get("--fit") in ("on", None) and _nonempty(v.get("--gpu-layers")):
        result.info(
            "--fit 处于开启状态，但已显式设置 -ngl；fit 不会覆盖显式参数，"
            "若显存不足请留空 -ngl 让 fit 决定。"
        )

    if str(v.get("--load-mode") or "") == "mlock":
        result.warn("--load-mode=mlock 会锁住内存中的模型，占用大量物理内存。")

    if str(v.get("--lazy-mode") or "") == "on":
        result.info("--lazy-mode=on 需要 mmap；在 Windows 上该功能可能被忽略。")


def _file_operands(flag: str, item: Any) -> List[str]:
    """File paths named by one value of a path-taking flag.

    ``--lora-scaled`` takes ``FNAME:SCALE[,FNAME:SCALE...]``.  The scale is
    split off at the *last* colon, because a Windows path such as
    ``D:/x.gguf:0.5`` has a drive-letter colon of its own.
    """
    text = str(item).strip()
    if not text:
        return []
    if flag != "--lora-scaled":
        return [text]
    out: List[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        head, sep, tail = part.rpartition(":")
        try:
            float(tail)
            is_scale = bool(sep) and bool(head)
        except ValueError:
            is_scale = False
        out.append(head if is_scale else part)
    return out


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    """Interpret a config value as a toggle.

    ``"none"`` is deliberately *truthy* here: it is a real value for options
    such as ``--spec-type`` and ``--tools`` rather than a way of saying "off".
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "0", "false", "no", "off")


# ------------------------------------------------------------------ preflight


def port_available(host: str, port: int, timeout: float = 0.35) -> bool:
    """True when *host*:*port* can be bound, i.e. nothing else is using it.

    A **bind** test is used rather than a connect probe.  On Windows a socket
    that is bound but not yet listening silently drops incoming SYNs, and a
    socket whose accept backlog is full does the same, so a connect probe cannot
    tell "free" from "occupied until it times out" - it just burns the whole
    timeout and answers "free" in either case.  Binding answers definitively and
    in microseconds, which matters because :func:`suggest_port` probes a range
    of candidates.

    *timeout* is accepted for call-site compatibility and is unused.
    """
    host = (host or "127.0.0.1").strip()
    # A server bound to a wildcard address occupies the port on every local
    # address, so probing loopback is the correct check.
    if host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"
    try:
        port = int(port)
    except (TypeError, ValueError):
        return False
    if not 0 < port < 65536:
        return False

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def suggest_port(host: str, start: int, limit: int = 50) -> int:
    """First free port at or after *start*."""
    for candidate in range(int(start), int(start) + limit):
        if port_available(host, candidate):
            return candidate
    return int(start)


#: Flags whose explicit presence is worth calling out in the launch summary.
HIGHLIGHT_FLAGS: Tuple[str, ...] = (
    "--model", "--ctx-size", "--gpu-layers", "--flash-attn", "--cache-type-k",
    "--cache-type-v", "--parallel", "--ubatch-size", "--batch-size", "--threads",
)

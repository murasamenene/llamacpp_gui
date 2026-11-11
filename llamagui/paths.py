"""Filesystem layout discovery.

The GUI lives inside the llama.cpp installation directory but must not assume a
fixed drive letter or folder name: the whole tree is meant to be copyable to
another machine, and the user may move it.  Everything is therefore resolved
relative to this file, with an explicit override file as the escape hatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

#: Binaries the GUI drives.  Two are required, the rest are optional features.
REQUIRED_BINARIES = ("llama-server.exe", "llama-cli.exe")
OPTIONAL_BINARIES = (
    "llama-bench.exe",
    "llama-tokenize.exe",
    "llama-fit-params.exe",
    "llama-gguf-split.exe",
    "llama-quantize.exe",
    "llama-mtmd-cli.exe",
    "ggml-rpc-server.exe",
    # Runtime DLLs that must sit next to the executables.
    "ggml-base.dll",
    "llama.dll",
    "llama-common.dll",
)

#: CUDA runtime pieces, checked only to explain a "no GPU" situation.
CUDA_LIBS = ("ggml-cuda.dll", "cublas64_13.dll")


@dataclass(frozen=True)
class Layout:
    """Resolved paths for one installation."""

    root: Path
    gui_dir: Path
    config_dir: Path
    presets_dir: Path
    cache_dir: Path
    logs_dir: Path
    bin_dir: Path
    help_override: Optional[Path]
    source: str

    # -------------------------------------------------------------- binaries

    def binary(self, name: str) -> Path:
        return self.bin_dir / name

    def has(self, name: str) -> bool:
        return (self.bin_dir / name).is_file()

    @property
    def missing_required(self) -> List[str]:
        return [n for n in REQUIRED_BINARIES if not self.has(n)]

    @property
    def missing_optional(self) -> List[str]:
        return [n for n in OPTIONAL_BINARIES if not self.has(n)]

    @property
    def has_cuda_libs(self) -> bool:
        return all(self.has(n) for n in CUDA_LIBS)

    def ensure_dirs(self) -> None:
        for d in (self.config_dir, self.presets_dir, self.cache_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def install_file(self) -> Path:
        return self.gui_dir / "install.json"

    def as_dict(self) -> Dict[str, str]:
        return {
            "root": str(self.root),
            "gui_dir": str(self.gui_dir),
            "config_dir": str(self.config_dir),
            "bin_dir": str(self.bin_dir),
            "source": self.source,
        }


def project_dir() -> Path:
    """Directory containing the ``llamagui`` package (i.e. ``.../gui``)."""
    return Path(__file__).resolve().parent.parent


def _candidate_roots() -> List[Path]:
    gui = project_dir()
    roots: List[Path] = []
    # 1. The GUI directory itself (binaries sit next to gui/ in the shipped
    #    layout, but a flat extraction may put everything in one folder).
    roots.append(gui)
    # 2. Its parent (the normal layout: F:\llama.cpp\gui -> F:\llama.cpp).
    roots.append(gui.parent)
    # 3. The current working directory, for "run from anywhere" launches.
    try:
        roots.append(Path.cwd())
    except OSError:
        pass
    # De-duplicate while preserving order.
    seen: List[Path] = []
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            continue
        if rp not in seen:
            seen.append(rp)
    return seen


def _has_server(directory: Path) -> bool:
    return (directory / "llama-server.exe").is_file()


def read_install_override(gui: Path) -> Optional[Path]:
    """Read an explicit root from ``gui/install.json`` if present."""
    path = gui / "install.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    root = data.get("root") if isinstance(data, dict) else None
    if isinstance(root, str) and root.strip():
        return Path(root).expanduser()
    return None


def read_help_dump(directory: Path) -> Optional[Path]:
    """The repo root ships ``_server_help.txt``; it is a useful offline snapshot."""
    path = directory / "_server_help.txt"
    return path if path.is_file() else None


def discover_root(gui_dir: Optional[Path] = None) -> tuple[Optional[Path], str]:
    """Locate the directory holding the llama.cpp binaries.

    Returns ``(root, source)`` where *source* explains how it was found, so the
    UI can show the user why it picked a particular folder.
    """
    gui = gui_dir or project_dir()

    override = read_install_override(gui)
    if override is not None:
        candidate = override if override.is_absolute() else (gui / override)
        if _has_server(candidate):
            return candidate.resolve(), "install.json"
        # An override that no longer exists is a hard signal something moved.
        return None, f"install.json 指向的目录里没有 llama-server.exe: {candidate}"

    for candidate in _candidate_roots():
        if _has_server(candidate):
            return candidate, "自动探测"

    return None, "未找到 llama-server.exe"


def layout(gui_dir: Optional[Path] = None) -> Layout:
    """Build the :class:`Layout`, including the config tree."""
    gui = gui_dir or project_dir()
    root, source = discover_root(gui)
    if root is None:
        # Degrade gracefully: keep the GUI usable so it can show the error and
        # let the user pick a folder, rather than failing at import time.
        root = gui
        source = source or "未找到"
    config_dir = gui / "config"
    return Layout(
        root=root,
        gui_dir=gui,
        config_dir=config_dir,
        presets_dir=config_dir / "presets",
        cache_dir=config_dir / "cache",
        logs_dir=config_dir / "logs",
        bin_dir=root,
        help_override=read_help_dump(root) or read_help_dump(gui),
        source=source,
    )


def write_install_override(gui_dir: Path, root: Path) -> Path:
    """Persist an explicit root so later launches do not have to guess."""
    path = gui_dir / "install.json"
    payload = {"root": str(root), "note": "llamagui 自动生成：llama.cpp 二进制目录"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def free_space_bytes(path: Path) -> int:
    """Free bytes on the volume holding *path* (0 when unavailable)."""
    import shutil

    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0

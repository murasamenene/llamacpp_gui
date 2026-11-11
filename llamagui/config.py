"""Configuration, presets, and recent-model bookkeeping.

State is stored as JSON under ``gui/config``:

``config.json``
    The live GUI state: argument values, window geometry, last model, options.
``presets/<name>.json``
    Named argument sets the user can switch between and share.
``cache/``
    Parsed help dumps and the device inventory, both safely regenerable.
``logs/``
    Raw launch logs, one per start.

Only explicitly-set arguments are stored.  A key whose value is ``None`` means
"not set by the user", which is what allows the GUI to omit it from the command
line so llama.cpp's own defaults (and ``--fit``) stay in charge.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .paths import Layout, layout as default_layout

#: Bumped when the on-disk shape changes in a way that needs migration.
CONFIG_VERSION = 1

_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


#: Config keys renamed by the GUI.  The stored value keeps its meaning (it is
#: the checkbox state), only the key changes.  ``--no-mmproj-auto`` used to be
#: the key of the "auto mmproj" toggle, which made a ticked box emit the flag
#: that *disables* the feature.
_RENAMED_KEYS = {"--no-mmproj-auto": "--mmproj-auto"}


def migrate_values(values: Dict[str, Any]) -> Dict[str, Any]:
    """Rename legacy keys in an argument mapping (config or preset)."""
    out = dict(values)
    for old, new in _RENAMED_KEYS.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    return out


def sanitize_name(name: str) -> str:
    """Make *name* safe to use as a file stem."""
    cleaned = _INVALID_NAME.sub("_", name).strip().strip(".")
    return cleaned or "preset"


@dataclass
class Config:
    """Live GUI configuration."""

    #: Canonical long flag -> user value.  Absent or ``None`` means unset.
    values: Dict[str, Any] = field(default_factory=dict)
    #: Environment variables to set for the launched process.
    env: Dict[str, str] = field(default_factory=dict)
    window: Dict[str, Any] = field(default_factory=dict)
    ui: Dict[str, Any] = field(default_factory=dict)
    #: Recently used local GGUF paths, most recent first.
    recent_models: List[str] = field(default_factory=list)
    #: Per-model argument overrides, keyed by absolute model path.
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    version: int = CONFIG_VERSION

    # ------------------------------------------------------------- accessors

    def get(self, flag: str, default: Any = None) -> Any:
        value = self.values.get(flag, default)
        return default if value is None else value

    def raw(self, flag: str) -> Any:
        return self.values.get(flag)

    def set(self, flag: str, value: Any) -> None:
        self.values[flag] = value

    def clear(self, flag: str) -> None:
        self.values.pop(flag, None)

    def update(self, mapping: Dict[str, Any], *, replace: bool = False) -> None:
        if replace:
            self.values = dict(mapping)
        else:
            self.values.update(mapping)

    @property
    def explicit_flags(self) -> List[str]:
        """Flags the user actually set to a value (``None`` excluded)."""
        return [f for f, v in self.values.items() if v is not None]

    def push_recent(self, path: str, limit: int = 12) -> None:
        path = str(path)
        if not path:
            return
        self.recent_models = [p for p in self.recent_models if p != path]
        self.recent_models.insert(0, path)
        del self.recent_models[limit:]

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "values": self.values,
            "env": self.env,
            "window": self.window,
            "ui": self.ui,
            "recent_models": self.recent_models,
            "model_overrides": self.model_overrides,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        if not isinstance(data, dict):
            return cls()
        values = data.get("values")
        env = data.get("env")
        window = data.get("window")
        ui = data.get("ui")
        recent = data.get("recent_models")
        overrides = data.get("model_overrides")
        return cls(
            values=migrate_values(values) if isinstance(values, dict) else {},
            env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
            window=dict(window) if isinstance(window, dict) else {},
            ui=dict(ui) if isinstance(ui, dict) else {},
            recent_models=[str(p) for p in recent] if isinstance(recent, list) else [],
            model_overrides=(
                {str(k): dict(v) for k, v in overrides.items() if isinstance(v, dict)}
                if isinstance(overrides, dict)
                else {}
            ),
            version=int(data.get("version") or CONFIG_VERSION),
        )


@dataclass
class Preset:
    """A named argument set."""

    name: str
    values: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    #: Optional model path this preset is meant for (informational).
    model: str = ""
    created: float = 0.0
    modified: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "values": self.values,
            "created": self.created or time.time(),
            "modified": self.modified or time.time(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        values = data.get("values")
        return cls(
            name=str(data.get("name") or "preset"),
            values=migrate_values(values) if isinstance(values, dict) else {},
            description=str(data.get("description") or ""),
            model=str(data.get("model") or ""),
            created=float(data.get("created") or 0.0),
            modified=float(data.get("modified") or 0.0),
        )


class Store:
    """Reads and writes the config tree."""

    def __init__(self, layout: Optional[Layout] = None) -> None:
        self.layout = layout or default_layout()

    # ------------------------------------------------------------------ core

    @property
    def config_path(self) -> Path:
        return self.layout.config_dir / "config.json"

    def load(self) -> Config:
        path = self.config_path
        if not path.is_file():
            return Config()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt config must never block startup: fall back to defaults
            # but preserve the bad file so the user can inspect it.
            try:
                path.replace(path.with_suffix(".json.corrupt"))
            except OSError:
                pass
            return Config()
        config = Config.from_dict(data)
        return self._migrate(config)

    def save(self, config: Config) -> None:
        self.layout.ensure_dirs()
        path = self.config_path
        payload = json.dumps(config.to_dict(), indent=2, ensure_ascii=False)
        # Write via a temporary file so an interrupted save cannot truncate the
        # existing configuration.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    def _migrate(self, config: Config) -> Config:
        """Apply forward migrations for older config files."""
        if config.version > CONFIG_VERSION:
            # A newer GUI wrote this; keep the data but do not interpret it.
            config.ui.setdefault("newer_config_warning", True)
            return config
        if config.version < 1:
            config.version = CONFIG_VERSION
        return config

    # ---------------------------------------------------------------- presets

    def preset_path(self, name: str) -> Path:
        return self.layout.presets_dir / f"{sanitize_name(name)}.json"

    def list_presets(self) -> List[Preset]:
        out: List[Preset] = []
        if not self.layout.presets_dir.is_dir():
            return out
        for path in sorted(self.layout.presets_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            preset = Preset.from_dict(data)
            if not preset.name:
                preset.name = path.stem
            out.append(preset)
        return out

    def load_preset(self, name: str) -> Optional[Preset]:
        path = self.preset_path(name)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return Preset.from_dict(data)

    def save_preset(self, preset: Preset) -> Path:
        self.layout.ensure_dirs()
        now = time.time()
        if not preset.created:
            preset.created = now
        preset.modified = now
        path = self.preset_path(preset.name)
        path.write_text(
            json.dumps(preset.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def delete_preset(self, name: str) -> bool:
        path = self.preset_path(name)
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False

    def rename_preset(self, old: str, new: str) -> Optional[Path]:
        preset = self.load_preset(old)
        if preset is None:
            return None
        preset.name = new
        path = self.save_preset(preset)
        if sanitize_name(old) != sanitize_name(new):
            self.delete_preset(old)
        return path

    # ------------------------------------------------------------------ cache

    def cache_path(self, name: str) -> Path:
        return self.layout.cache_dir / name

    def read_cache(self, name: str) -> Optional[str]:
        path = self.cache_path(name)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def write_cache(self, name: str, text: str) -> None:
        self.layout.ensure_dirs()
        try:
            self.cache_path(name).write_text(text, encoding="utf-8")
        except OSError:
            pass

    def read_cache_json(self, name: str) -> Optional[Any]:
        raw = self.read_cache(name)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def write_cache_json(self, name: str, payload: Any) -> None:
        try:
            self.write_cache(name, json.dumps(payload, indent=2, ensure_ascii=False))
        except (TypeError, ValueError):
            pass

    def log_path(self, prefix: str, when: Optional[float] = None) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when or time.time()))
        self.layout.ensure_dirs()
        return self.layout.logs_dir / f"{prefix}-{stamp}.txt"

    # ---------------------------------------------------------------- exports

    def export_preset(self, preset: Preset, target: Path) -> None:
        target.write_text(
            json.dumps(preset.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def import_preset(self, source: Path) -> Preset:
        data = json.loads(source.read_text(encoding="utf-8"))
        preset = Preset.from_dict(data)
        if not preset.name:
            preset.name = source.stem
        return preset


# ---------------------------------------------------------------- merging


def merge_values(
    base: Dict[str, Any], overlay: Dict[str, Any], *, keys: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """Overlay *overlay* onto *base*, dropping ``None`` entries.

    ``None`` in *overlay* means "not specified", so it must not clobber an
    inherited value.  Passing *keys* restricts which overlay entries apply,
    which is how the UI merges only the fields the user edited.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if keys is not None and key not in keys:
            continue
        if value is None:
            continue
        result[key] = value
    return result


def presets_summary(presets: Iterable[Preset]) -> List[str]:
    """Human readable one-line summaries, for list boxes."""
    out: List[str] = []
    for p in presets:
        bits = []
        if p.model:
            bits.append(os.path.basename(p.model))
        if p.description:
            bits.append(p.description)
        out.append(f"{p.name}" + (f"  -  {' / '.join(bits)}" if bits else ""))
    return out

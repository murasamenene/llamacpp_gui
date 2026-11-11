"""llamagui - a zero-dependency tkinter launcher for the local llama.cpp build.

Pure Python standard library only.  No pip dependencies, no venv, no compiled
extensions.  The package drives the release binaries that live next to it
(``llama-server.exe`` and friends) and adds:

* a full parameter surface generated from the local binaries' own ``--help``
  output, cross-checked against a curated metadata table,
* live process output with structured log handling,
* GGUF header inspection for pre-flight VRAM / context estimation,
* a one-click launch of the bundled llama-server web UI.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

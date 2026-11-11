"""``python -m llamagui`` entry point."""

from __future__ import annotations

import sys


def _run() -> int:
    # Imported lazily so that ``python -m llamagui --version`` style probes do
    # not need a display.
    from .app import main

    return main()


if __name__ == "__main__":
    sys.exit(_run())

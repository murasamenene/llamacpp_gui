#!/usr/bin/env pythonw
"""llama.cpp 启动器 —— 双击这个文件即可运行。

使用 ``pythonw.exe`` 运行（扩展名 ``.pyw``）可以避免弹出黑色控制台窗口。
如果关联的是 ``python.exe``，仍然能工作，只是会多一个控制台窗口。

命令行等价形式::

    pythonw llamagui.pyw
    python -m llamagui
"""

from __future__ import annotations

import os
import sys
import traceback

# Make ``llamagui`` importable when this file is launched directly, regardless
# of the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _fatal(message: str) -> None:
    """Report a startup failure without a console to print to."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("llama.cpp 启动器", message)
        root.destroy()
    except Exception:  # noqa: BLE001 - last resort
        sys.stderr.write(message + "\n")


def main() -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        _fatal(
            "这个 Python 没有可用的 tkinter，无法运行图形界面。\n\n"
            f"{exc}\n\n"
            "请安装带 tkinter 的官方 Python 发行版，"
            "或在“应用和功能”里为该 Python 添加 tcl/tk 组件。"
        )
        return 1

    try:
        from llamagui.app import main as app_main

        return app_main()
    except Exception:  # noqa: BLE001 - surface a traceback in a dialog
        _fatal(
            "启动失败，发生未预期的错误：\n\n"
            + traceback.format_exc(limit=12)
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

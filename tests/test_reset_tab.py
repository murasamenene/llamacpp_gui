"""The "恢复本页默认" button: clears only the visible tab's parameters."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

GUI_DIR = Path(__file__).resolve().parent.parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from tests.test_core import temp_layout  # noqa: E402
from tests.test_ui_smoke import _DISPLAY_REASON, _HAS_DISPLAY  # noqa: E402


@unittest.skipUnless(_HAS_DISPLAY, f"没有可用的图形显示：{_DISPLAY_REASON}")
class ResetCurrentTabTests(unittest.TestCase):
    def setUp(self) -> None:
        from llamagui import paths
        from llamagui.app import App

        if paths.layout().missing_required:
            self.skipTest("缺少 llama-server.exe")
        self.tmp = tempfile.TemporaryDirectory(prefix="llamagui-reset-")
        self.app = App(temp_layout(Path(self.tmp.name)))
        self.app.root.withdraw()
        self.model = Path(self.tmp.name) / "m.gguf"
        self.model.write_bytes(b"GGUF")

    def tearDown(self) -> None:
        try:
            self.app.shutdown()
            self.app.root.destroy()
        except Exception:  # noqa: BLE001
            pass
        self.tmp.cleanup()

    def pump(self, seconds: float = 0.3) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.root.update()
            time.sleep(0.01)

    def select(self, tab) -> None:  # noqa: ANN001
        self.app.notebook.select(tab)
        self.pump(0.2)

    def field(self, tab, flag: str):  # noqa: ANN001
        return next(f for f in tab.fields if f.spec.flag == flag)

    def test_clears_only_the_visible_tab(self) -> None:
        app = self.app
        app.set_value("--model", str(self.model))
        app.set_value("--temp", 0.7)
        app.set_value("--top-k", 20)
        app.set_value("--ctx-size", 8192)
        app.sampling_tab.refresh()
        self.select(app.sampling_tab)

        with mock.patch("llamagui.app.messagebox.askyesno", return_value=True) as ask:
            cleared = app.reset_current_tab()
        ask.assert_called_once()
        self.assertEqual(set(cleared), {"--temp", "--top-k"})
        self.assertNotIn("--temp", app.config.values)
        self.assertNotIn("--top-k", app.config.values)
        # Other tabs are untouched.
        self.assertEqual(app.config.values.get("--ctx-size"), 8192)
        self.assertEqual(app.config.values.get("--model"), str(self.model))
        # Field shows "not set", and nothing reaches the command line.
        self.assertEqual(self.field(app.sampling_tab, "--temp").var.get(), "")
        argv = app.build_command().argv
        self.assertNotIn("--temp", argv)
        self.assertNotIn("--top-k", argv)
        self.assertIn("--ctx-size", argv)

    def test_cancel_changes_nothing(self) -> None:
        app = self.app
        app.set_value("--temp", 0.7)
        self.select(app.sampling_tab)
        with mock.patch("llamagui.app.messagebox.askyesno", return_value=False):
            self.assertEqual(app.reset_current_tab(), [])
        self.assertEqual(app.config.values.get("--temp"), 0.7)

    def test_pending_debounced_edit_is_also_cleared(self) -> None:
        app = self.app
        self.select(app.sampling_tab)
        field = self.field(app.sampling_tab, "--temp")
        field.var.set("0.3")  # typed but not yet past the debounce
        with mock.patch("llamagui.app.messagebox.askyesno", return_value=True):
            cleared = app.reset_current_tab()
        self.assertIn("--temp", cleared)
        self.assertIsNone(app.get_value("--temp"))
        self.pump(1.0)  # a late debounce must not resurrect the value
        self.assertIsNone(app.get_value("--temp"))

    def test_model_tab_keeps_model_and_repo(self) -> None:
        app = self.app
        app.set_value("--model", str(self.model))
        app.set_value("--hf-repo", "user/repo")
        app.set_value("--jinja", False)
        self.select(app.model_tab)
        with mock.patch("llamagui.app.messagebox.askyesno", return_value=True):
            cleared = app.reset_current_tab()
        self.assertIn("--jinja", cleared)
        self.assertNotIn("--jinja", app.config.values)
        self.assertEqual(app.config.values.get("--model"), str(self.model))
        self.assertEqual(app.config.values.get("--hf-repo"), "user/repo")

    def test_nothing_set_asks_nothing(self) -> None:
        app = self.app
        app.config.values.clear()
        self.select(app.context_tab)
        with mock.patch("llamagui.app.messagebox.askyesno") as ask:
            self.assertEqual(app.reset_current_tab(), [])
        ask.assert_not_called()

    def test_button_disabled_on_tabs_without_parameters(self) -> None:
        app = self.app
        self.select(app.console_tab)
        self.assertEqual(str(app.reset_tab_button.cget("state")), "disabled")
        self.select(app.api_tab)
        self.assertEqual(str(app.reset_tab_button.cget("state")), "disabled")
        self.select(app.hardware_tab)
        self.assertEqual(str(app.reset_tab_button.cget("state")), "normal")

    def test_bench_tab_resets_form_but_keeps_model(self) -> None:
        app = self.app
        bench = app.bench_tab
        bench.model_var.set(str(self.model))
        bench.prompt_var.set("2048")
        bench.ngl_var.set("99")
        self.select(bench)
        with mock.patch("llamagui.app.messagebox.askyesno", return_value=True):
            app.reset_current_tab()
        self.assertEqual(bench.prompt_var.get(), "512")
        self.assertEqual(bench.ngl_var.get(), "auto")
        self.assertEqual(bench.model_var.get(), str(self.model))


if __name__ == "__main__":
    unittest.main(verbosity=2)

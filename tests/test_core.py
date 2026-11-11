"""Unit tests for the static layer: parsing, estimation, config, and command building.

These tests never load a model and never start llama-server.  They may read the
help snapshot that ships with the installation (``_server_help.txt``) because
that is what the GUI itself parses at first launch.

Run from the ``gui`` directory::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent.parent
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from llamagui import argspec, cmdline, config as config_mod, estimate, gguf, helpparse, logparse  # noqa: E402

ROOT = GUI_DIR.parent
HELP_SNAPSHOT = ROOT / "_server_help.txt"


# --------------------------------------------------------------------- helpers


def build_gguf(
    path: Path,
    metadata: dict,
    tensors,
    *,
    version: int = 3,
    alignment: int = 32,
) -> Path:
    """Write a minimal but structurally valid GGUF file for parser tests."""
    type_ids = {
        "uint32": (4, "<I"),
        "int32": (5, "<i"),
        "uint64": (10, "<Q"),
        "int64": (11, "<q"),
        "float32": (6, "<f"),
        "float64": (12, "<d"),
        "bool": (7, "<?"),
        "string": (8, None),
    }

    def write_value(buf: bytearray, value, kind: str) -> None:
        if kind == "array":
            raise AssertionError("array values must be written via write_array")
        tid, fmt = type_ids[kind]
        if kind == "string":
            raw = str(value).encode("utf-8")
            buf += struct.pack("<I", tid)
            buf += struct.pack("<Q", len(raw))
            buf += raw
            return
        if kind == "bool":
            buf += struct.pack("<I", tid)
            buf += struct.pack("<?", bool(value))
            return
        if kind == "uint64":
            # Arrays encode their element type as a u32, matching GGUF.
            buf += struct.pack("<I", tid)
            buf += struct.pack("<Q", int(value))
            return
        buf += struct.pack("<I", tid)
        buf += struct.pack(fmt, value)

    def write_array(buf: bytearray, values, elem_kind: str) -> None:
        tid, _ = type_ids[elem_kind]
        buf += struct.pack("<I", 9)  # GGUF_TYPE_ARRAY
        buf += struct.pack("<I", tid)
        buf += struct.pack("<Q", len(values))
        for value in values:
            if elem_kind == "string":
                raw = str(value).encode("utf-8")
                buf += struct.pack("<Q", len(raw))
                buf += raw
            else:
                _, fmt = type_ids[elem_kind]
                buf += struct.pack(fmt, value)

    with path.open("wb") as fh:
        fh.write(struct.pack("<I", gguf.MAGIC))
        fh.write(struct.pack("<I", version))
        fh.write(struct.pack("<Q", len(tensors)))
        fh.write(struct.pack("<Q", len(metadata) + 1))

        # general.alignment first, as real files do.
        key = b"general.alignment"
        fh.write(struct.pack("<Q", len(key)))
        fh.write(key)
        fh.write(struct.pack("<I", 4))
        fh.write(struct.pack("<I", alignment))

        for name, spec_value in metadata.items():
            kind, value = spec_value
            raw = name.encode("utf-8")
            fh.write(struct.pack("<Q", len(raw)))
            fh.write(raw)
            buf = bytearray()
            if kind == "array":
                elem_kind, items = value
                write_array(buf, items, elem_kind)
            else:
                write_value(buf, value, kind)
            fh.write(bytes(buf))

        for name, dims, ggml_type in tensors:
            raw = name.encode("utf-8")
            fh.write(struct.pack("<Q", len(raw)))
            fh.write(raw)
            fh.write(struct.pack("<I", len(dims)))
            for d in dims:
                fh.write(struct.pack("<Q", d))
            fh.write(struct.pack("<I", ggml_type))
            fh.write(struct.pack("<Q", 0))

        # Pad descriptors to the alignment and leave the data area empty: the
        # parser only ever reads the header.
        while fh.tell() % alignment:
            fh.write(b"\x00")
    return path


def temp_layout(config_dir: Path):
    """A layout pointing at a throwaway config tree.

    The UI tests need the real llama.cpp binaries but must never read or write
    the user's actual ``gui/config``: a test that closes the window would
    otherwise overwrite their saved settings.
    """
    from llamagui import paths

    real = paths.layout()
    return paths.Layout(
        root=real.root,
        gui_dir=real.gui_dir,
        config_dir=config_dir,
        presets_dir=config_dir / "presets",
        cache_dir=config_dir / "cache",
        logs_dir=config_dir / "logs",
        bin_dir=real.bin_dir,
        help_override=real.help_override,
        source="test",
    )


def typical_metadata(    *,
    arch: str = "llama",
    n_layer: int = 32,
    n_head: int = 32,
    n_head_kv: int = 8,
    n_embd: int = 4096,
    n_ctx_train: int = 8192,
    quant: int = 12,
    extra: dict | None = None,
) -> dict:
    md = {
        "general.architecture": ("string", arch),
        "general.name": ("string", f"test-{arch}"),
        f"{arch}.block_count": ("uint32", n_layer),
        f"{arch}.attention.head_count": ("uint32", n_head),
        f"{arch}.attention.head_count_kv": ("uint32", n_head_kv),
        f"{arch}.embedding_length": ("uint32", n_embd),
        f"{arch}.context_length": ("uint32", n_ctx_train),
        "tokenizer.ggml.tokens": ("array", ("string", ["a", "b", "c", "d"])),
    }
    if extra:
        md.update(extra)
    return md


# ------------------------------------------------------------------ helpparse


class LauncherScriptTests(unittest.TestCase):
    """Guard the launcher scripts against the bugs that are hardest to notice.

    ``start.bat`` must stay **pure ASCII with CRLF line endings**.  cmd.exe
    parses batch files by byte offset and loses its position when the file
    contains multi-byte characters, after which it executes fragments of comment
    lines as commands.  Observed on a GBK console::

        'm' is not recognized as an internal or external command
        '--------------' is not recognized as an internal or external command

    A fragment is usually inert, but it could equally be ``del`` or ``start``,
    and the failure is invisible from Python - only the console shows it.  These
    assertions are cheap and prevent that whole class of regression.
    """

    BAT = GUI_DIR / "start.bat"

    def test_start_bat_is_pure_ascii(self) -> None:
        raw = self.BAT.read_bytes()
        offending = [(i, b) for i, b in enumerate(raw) if b > 127]
        self.assertEqual(
            offending[:5], [],
            f"start.bat 含有 {len(offending)} 个非 ASCII 字节；"
            "cmd.exe 会因此解析错位并把注释片段当命令执行。"
            "请把所有非英文说明移到 README.md。",
        )

    def test_start_bat_has_no_bom(self) -> None:
        raw = self.BAT.read_bytes()
        self.assertNotEqual(raw[:3], b"\xef\xbb\xbf", "start.bat 不应有 UTF-8 BOM")

    def test_start_bat_uses_crlf_endings(self) -> None:
        raw = self.BAT.read_bytes()
        self.assertNotIn(b"\r\r", raw)
        lone_lf = raw.replace(b"\r\n", b"").count(b"\n")
        self.assertEqual(lone_lf, 0, "start.bat 存在单独的 LF 换行，应统一为 CRLF")

    def test_start_bat_prefers_windowed_interpreters(self) -> None:
        """A console window must not appear, so pyw/pythonw come first."""
        text = self.BAT.read_text(encoding="ascii")
        pyw_at = text.find("pyw -3 -c")
        pythonw_at = text.find("pythonw -c")
        py_at = text.find("py -3 -c")
        python_at = text.find("python -c")
        self.assertNotEqual(pyw_at, -1, "未找到 pyw 分支")
        self.assertNotEqual(pythonw_at, -1, "未找到 pythonw 分支")
        self.assertLess(pyw_at, pythonw_at, "pyw 应优先于 pythonw")
        if py_at != -1 and python_at != -1:
            self.assertLess(pythonw_at, py_at, "有窗口的解释器必须排在会弹控制台的前面")

    def test_start_bat_checks_tkinter_before_launching(self) -> None:
        text = self.BAT.read_text(encoding="ascii")
        self.assertIn("import tkinter", text)
        self.assertIn("llamagui.pyw", text)

    def test_pyw_entry_point_is_ascii_safe(self) -> None:
        """The .pyw may contain any UTF-8 (Python reads it), but must exist."""
        entry = GUI_DIR / "llamagui.pyw"
        source = entry.read_text(encoding="utf-8")
        self.assertIn("def main()", source)
        # It must be importable as UTF-8 and must not require a console.
        self.assertIn("pythonw", source.lower())

    def test_explains_the_ascii_requirement_in_the_file(self) -> None:
        """A future editor must be warned before re-adding translated comments."""
        text = self.BAT.read_text(encoding="ascii").lower()
        self.assertIn("ascii", text)
        self.assertIn("cmd.exe", text)


@unittest.skipUnless(HELP_SNAPSHOT.is_file(), "help snapshot not present")
class HelpParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = HELP_SNAPSHOT.read_text(encoding="utf-8", errors="replace")
        cls.doc = helpparse.parse_help(cls.text, "llama-server")

    def test_recovers_essentially_every_option(self) -> None:
        self.assertGreaterEqual(len(self.doc.options), 250)
        # The parser should understand every definition line in a real dump.
        self.assertEqual(self.doc.unparsed, [])

    def test_sections_detected(self) -> None:
        self.assertIn("common params", self.doc.sections)
        self.assertIn("sampling params", self.doc.sections)
        self.assertIn("speculative params", self.doc.sections)

    def test_short_flags(self) -> None:
        cases = {
            "--ctx-size": "-c",
            "--threads": "-t",
            "--gpu-layers": "-ngl",
            "--flash-attn": "-fa",
            "--ubatch-size": "-ub",
            "--override-tensor": "-ot",
            "--model": "-m",
        }
        for long, short in cases.items():
            with self.subTest(long=long):
                opt = self.doc.get(long)
                self.assertIsNotNone(opt, long)
                self.assertIn(short, opt.shorts)

    def test_hyphenated_negated_short_flags(self) -> None:
        # These are internal negated toggles written with a single dash; they
        # were the reason a naive short-flag regex silently lost options.
        kv = self.doc.get("--kv-unified")
        self.assertIsNotNone(kv)
        self.assertIn("-no-kvu", kv.shorts)
        self.assertIn("--no-kv-unified", kv.aliases)

        agent = self.doc.get("--agent")
        self.assertIsNotNone(agent)
        self.assertIn("-no-ag", agent.shorts)

    def test_choices_from_usage_enum(self) -> None:
        fa = self.doc.get("--flash-attn")
        self.assertEqual(set(fa.choices), {"on", "off", "auto"})
        rope = self.doc.get("--rope-scaling")
        self.assertEqual(set(rope.choices), {"none", "linear", "yarn"})

    def test_defaults_are_not_borrowed_from_the_next_option(self) -> None:
        # --no-host is documented without a default; --cache-type-k follows it
        # and used to leak its "allowed values" into --no-host.
        nohost = self.doc.get("--no-host")
        self.assertIsNotNone(nohost)
        self.assertNotEqual(nohost.choices, ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1"))
        ctk = self.doc.get("--cache-type-k")
        self.assertIn("q8_0", ctk.choices)
        self.assertEqual(ctk.default_text, "f16")

    def test_env_extraction(self) -> None:
        self.assertEqual(self.doc.get("--threads").env, "LLAMA_ARG_THREADS")
        self.assertEqual(self.doc.get("--ctx-size").env, "LLAMA_ARG_CTX_SIZE")

    def test_aliases_are_addressable(self) -> None:
        for alias in ("--n-predict", "--usage", "--hf-repo", "--webui"):
            with self.subTest(alias=alias):
                self.assertTrue(self.doc.has(alias))

    def test_version_banner_helpers(self) -> None:
        info = "0.4.1-dev (build 11026, commit b49650adb)"
        self.assertEqual(helpparse.build_number(info), "11026")
        self.assertEqual(helpparse.build_commit(info), "b49650adb")

    def test_every_table_flag_exists_in_this_build(self) -> None:
        missing = sorted(f for f in argspec.SPECS if not self.doc.has(f))
        self.assertEqual(missing, [], f"argspec 中的参数在本 build 不存在: {missing}")

    def test_every_negative_flag_exists_in_this_build(self) -> None:
        missing = sorted(
            f"{spec.flag}->{spec.neg}"
            for spec in argspec.SPECS.values()
            if spec.neg and not self.doc.has(spec.neg)
        )
        self.assertEqual(missing, [])

    def test_table_env_names_match_help(self) -> None:
        mismatches = []
        for spec in argspec.SPECS.values():
            if not spec.env:
                continue
            opt = self.doc.get(spec.flag)
            if opt is not None and opt.env and opt.env != spec.env:
                mismatches.append((spec.flag, spec.env, opt.env))
        self.assertEqual(mismatches, [])


class HelpParseSyntheticTests(unittest.TestCase):
    """Formatting details the real dump may not exercise."""

    def test_column_based_description_split(self) -> None:
        usage, desc = helpparse.split_usage_line(
            "-t,    --threads N                      number of CPU threads"
        )
        self.assertEqual(usage, "-t,    --threads N")
        self.assertEqual(desc, "number of CPU threads")

    def test_long_usage_line_has_no_description(self) -> None:
        line = "-ot,   --override-tensor <tensor name pattern>=<buffer type>,..."
        usage, desc = helpparse.split_usage_line(line)
        self.assertEqual(desc, "")
        self.assertIn("--override-tensor", usage)

    def test_continuation_lines_are_not_options(self) -> None:
        doc = helpparse.parse_help(
            "----- common params -----\n"
            "\n"
            "-c,    --ctx-size N                     size of the context\n"
            "                                        (default: 0)\n"
            "                                        (env: LLAMA_ARG_CTX_SIZE)\n"
            "--metrics                               enable metrics\n"
            "                                        (env: LLAMA_ARG_ENDPOINT_METRICS)\n"
            "\n"
            "----- sampling params -----\n"
            "\n"
            "--temp, --temperature N                 temperature (default: 0.80)\n"
        )
        self.assertIn("--ctx-size", doc.options)
        self.assertIn("--metrics", doc.options)
        self.assertIn("--temperature", doc.options)
        self.assertEqual(doc.get("--ctx-size").default_text, "0")
        self.assertEqual(doc.get("--metrics").env, "LLAMA_ARG_ENDPOINT_METRICS")

    def test_option_with_no_description_line(self) -> None:
        doc = helpparse.parse_help("--offline\n")
        self.assertIn("--offline", doc.options)

    def test_parse_version(self) -> None:
        self.assertEqual(
            helpparse.parse_version("version: 1.2.3\nbuilt with gcc"),
            "1.2.3",
        )
        self.assertEqual(helpparse.parse_version("no banner here"), "")


# ------------------------------------------------------------------------ gguf


class GgufTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parses_metadata_and_tensors(self) -> None:
        tensors = [
            ("token_embd.weight", (4096, 32000), 12),   # q4_k
            ("blk.0.attn_q.weight", (4096, 4096), 1),   # f16
        ]
        path = build_gguf(
            self.dir / "m.gguf",
            typical_metadata(quant=12),
            tensors,
        )
        info = gguf.load(str(path))
        self.assertIsNone(info.error)
        self.assertEqual(info.version, 3)
        self.assertEqual(info.tensor_count, 2)
        self.assertEqual(info.architecture, "llama")
        self.assertEqual(info.arch_get_int("block_count"), 32)
        self.assertEqual(info.arch_get_int("attention.head_count_kv"), 8)
        self.assertEqual(len(info.get_array("tokenizer.ggml.tokens")), 4)
        self.assertEqual([t.name for t in info.tensors][0], "token_embd.weight")
        self.assertGreater(info.data_offset, 0)
        self.assertEqual(info.data_offset % info.alignment, 0)

    def test_tensor_sizes_match_ggml_blocks(self) -> None:
        tensors = [
            ("a", (4096, 4096), 1),    # f16: 2 bytes/element
            ("b", (256, 256), 12),     # q4_k: 144 bytes per 256 elements
        ]
        path = build_gguf(self.dir / "s.gguf", typical_metadata(), tensors)
        info = gguf.load(str(path))
        by_name = {t.name: t for t in info.tensors}
        self.assertEqual(by_name["a"].n_bytes, 4096 * 4096 * 2)
        self.assertEqual(by_name["b"].n_bytes, (256 * 256 // 256) * 144)

    def test_bad_magic_is_reported_not_raised(self) -> None:
        path = self.dir / "junk.gguf"
        path.write_bytes(b"NOTGGUF" + b"\x00" * 64)
        info = gguf.load(str(path))
        self.assertIsNotNone(info.error)
        self.assertIn("magic", info.error.lower())

    def test_truncated_file_is_reported(self) -> None:
        path = self.dir / "short.gguf"
        path.write_bytes(struct.pack("<I", gguf.MAGIC) + struct.pack("<I", 3))
        info = gguf.load(str(path))
        self.assertIsNotNone(info.error)

    def test_missing_file_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            gguf.load(str(self.dir / "nope.gguf"))

    def test_geometry_extraction(self) -> None:
        path = build_gguf(self.dir / "g.gguf", typical_metadata(), [])
        info = gguf.load(str(path))
        geo = gguf.geometry_from_info(info)
        self.assertEqual(geo.n_layer, 32)
        self.assertEqual(geo.n_head, 32)
        self.assertEqual(geo.n_head_kv, 8)
        self.assertEqual(geo.n_embd, 4096)
        self.assertEqual(geo.head_dim_k, 128)
        self.assertEqual(geo.head_dim_v, 128)
        self.assertEqual(geo.n_ctx_train, 8192)
        self.assertEqual(geo.n_vocab, 4)
        self.assertFalse(geo.is_moe)

    def test_explicit_key_value_lengths_win(self) -> None:
        md = typical_metadata(extra={
            "llama.attention.key_length": ("uint32", 64),
            "llama.attention.value_length": ("uint32", 64),
        })
        path = build_gguf(self.dir / "k.gguf", md, [])
        geo = gguf.geometry_from_info(gguf.load(str(path)))
        self.assertEqual(geo.head_dim_k, 64)
        self.assertEqual(geo.head_dim_v, 64)

    def test_shard_name_parsing(self) -> None:
        self.assertEqual(
            gguf.parse_shard_name("Model-00002-of-00005.gguf"), (2, 5)
        )
        self.assertIsNone(gguf.parse_shard_name("Model.gguf"))

    def test_mmproj_detection(self) -> None:
        self.assertTrue(gguf.is_mmproj("mmproj-F16.gguf"))
        self.assertTrue(gguf.is_mmproj("MMPROJ.gguf"))
        self.assertFalse(gguf.is_mmproj("model.gguf"))


# -------------------------------------------------------------------- estimate


class EstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _info(self, **kwargs) -> gguf.GGUFInfo:
        path = build_gguf(self.dir / f"e{len(list(self.dir.iterdir()))}.gguf",
                          typical_metadata(**kwargs), [])
        return gguf.load(str(path))

    def test_kv_bytes_match_the_closed_form(self) -> None:
        info = self._info()
        geo = gguf.geometry_from_info(info)
        # 32 layers * 8192 ctx * 8 kv heads * (128 + 128) dims * 2 bytes
        expected = 32 * 8192 * 8 * 256 * 2
        got = estimate.kv_cache_bytes(geo, 8192, "f16", "f16")
        self.assertEqual(got, expected)

    def test_q8_kv_cache_is_smaller_than_f16(self) -> None:
        info = self._info()
        geo = gguf.geometry_from_info(info)
        f16 = estimate.kv_cache_bytes(geo, 8192, "f16", "f16")
        q8 = estimate.kv_cache_bytes(geo, 8192, "q8_0", "q8_0")
        self.assertLess(q8, f16)
        self.assertAlmostEqual(q8 / f16, (34 / 32) / 2, places=6)

    def test_kv_scales_linearly_with_context(self) -> None:
        info = self._info()
        geo = gguf.geometry_from_info(info)
        a = estimate.kv_cache_bytes(geo, 4096, "f16", "f16")
        b = estimate.kv_cache_bytes(geo, 8192, "f16", "f16")
        self.assertEqual(b, a * 2)

    def test_no_kv_offload_excludes_cache_from_vram(self) -> None:
        info = self._info()
        geo = gguf.geometry_from_info(info)
        self.assertEqual(
            estimate.kv_cache_bytes(geo, 8192, "f16", "f16", kv_offloaded=False), 0
        )

    def test_non_unified_kv_multiplies_by_slots(self) -> None:
        info = self._info()
        geo = gguf.geometry_from_info(info)
        one = estimate.kv_cache_bytes(geo, 4096, "f16", "f16", n_parallel=1, kv_unified=True)
        four = estimate.kv_cache_bytes(geo, 4096, "f16", "f16", n_parallel=4, kv_unified=False)
        self.assertEqual(four, one * 4)

    def test_sliding_window_shrinks_cache(self) -> None:
        md = typical_metadata(extra={"llama.attention.sliding_window": ("uint32", 1024)})
        path = build_gguf(self.dir / "swa.gguf", md, [])
        geo = gguf.geometry_from_info(gguf.load(str(path)))
        full = estimate.kv_cache_bytes(geo, 8192, "f16", "f16", swa_full=True)
        windowed = estimate.kv_cache_bytes(geo, 8192, "f16", "f16", swa_full=False)
        self.assertLess(windowed, full)
        # All but the last layer are windowed.
        self.assertAlmostEqual(windowed / full, (31 * 1024 + 8192) / (32 * 8192), places=5)

    def test_moe_and_mla_detection(self) -> None:
        md = typical_metadata(extra={"llama.expert_count": ("uint32", 8)})
        path = build_gguf(self.dir / "moe.gguf", md, [])
        geo = gguf.geometry_from_info(gguf.load(str(path)))
        self.assertTrue(geo.is_moe)
        self.assertEqual(geo.n_expert, 8)

        mla_md = typical_metadata(extra={
            "llama.attention.kv_lora_rank": ("uint32", 512),
            "llama.rope.dimension_count": ("uint32", 64),
        })
        path = build_gguf(self.dir / "mla.gguf", mla_md, [])
        geo = gguf.geometry_from_info(gguf.load(str(path)))
        self.assertTrue(geo.is_mla)

    def test_estimate_fits_and_headroom(self) -> None:
        path = build_gguf(
            self.dir / "big.gguf",
            typical_metadata(),
            [("tok.weight", (4096, 32000), 12)],
        )
        info = gguf.load(str(path))
        free = 8 * estimate.GiB
        est = estimate.estimate(
            info, n_ctx=8192, device_total_bytes=20 * estimate.GiB,
            device_free_bytes=free,
        )
        self.assertEqual(est.total_bytes, est.weights_bytes + est.kv_bytes + est.compute_bytes)
        self.assertIn(est.fits, (True, False))
        self.assertAlmostEqual(est.headroom_gib, (free - est.total_bytes) / estimate.GiB, places=6)

    def test_max_context_is_capped_by_training_context(self) -> None:
        info = self._info(n_ctx_train=8192)
        est = estimate.estimate(
            info, n_ctx=8192, device_free_bytes=200 * estimate.GiB,
        )
        self.assertLessEqual(est.max_ctx_fit, 8192)

    def test_max_context_grows_with_free_vram(self) -> None:
        info = self._info()
        small = estimate.estimate(info, device_free_bytes=6 * estimate.GiB)
        large = estimate.estimate(info, device_free_bytes=18 * estimate.GiB)
        self.assertGreaterEqual(large.max_ctx_fit, small.max_ctx_fit)

    def test_weights_exceeding_vram_reports_no_context(self) -> None:
        path = build_gguf(
            self.dir / "huge.gguf",
            typical_metadata(),
            [("w", (8192, 8192), 1)],  # 128 MiB of f16 plus overhead
        )
        info = gguf.load(str(path))
        est = estimate.estimate(info, n_ctx=8192, device_free_bytes=16 * estimate.MiB)
        self.assertEqual(est.max_ctx_fit, 0)
        self.assertTrue(est.warnings)

    def test_partial_offload_scales_weights(self) -> None:
        path = build_gguf(
            self.dir / "p.gguf",
            typical_metadata(n_layer=40),
            [("w", (4096, 4096), 1)],
        )
        info = gguf.load(str(path))
        all_layers = estimate.estimate(info, gpu_layers="all", n_ctx=4096)
        half = estimate.estimate(info, gpu_layers="20", n_ctx=4096)
        self.assertLess(half.weights_bytes, all_layers.weights_bytes)
        self.assertAlmostEqual(half.weights_bytes / all_layers.weights_bytes, 0.5, places=5)

    def test_context_exceeding_training_context_warns(self) -> None:
        info = self._info(n_ctx_train=8192)
        est = estimate.estimate(info, n_ctx=32768, device_free_bytes=40 * estimate.GiB)
        self.assertTrue(any("训练上下文" in w for w in est.warnings))

    def test_suggest_context_returns_zero_without_vram(self) -> None:
        info = self._info()
        self.assertEqual(estimate.suggest_context(info, device_free_bytes=0), 0)

    def test_max_context_on_a_realistic_7b_shape(self) -> None:
        """Pin the sizing maths against a plausible 7B Q4 model.

        Tied embeddings are left out of this fixture, so the numbers are not the
        real 7B ones - what is pinned is the *arithmetic*: KV per token, the
        weights term, and the resulting context ladder.
        """
        path = build_gguf(
            self.dir / "7b.gguf",
            typical_metadata(n_layer=32, n_head=32, n_head_kv=8, n_embd=4096,
                             n_ctx_train=32768),
            [
                ("token_embd.weight", (4096, 152064), 12),  # q4_k
                ("output.weight", (4096, 152064), 14),      # q6_k
            ],
        )
        info = gguf.load(str(path))
        geo = gguf.geometry_from_info(info)

        # Per token in one layer: n_kv_head * (k_dim + v_dim) * 2 bytes
        #                            8 * (  128  +  128 ) * 2  = 2048 B
        # head_dim is inferred as n_embd / n_head because the fixture does not
        # set attention.key_length / value_length explicitly.
        self.assertEqual(geo.head_dim_k, 128)
        self.assertEqual(geo.head_dim_v, 128)
        self.assertEqual(geo.kv_per_token_bytes, 2048)
        self.assertEqual(estimate.KV_DTYPE_BYTES["f16"], (2.0, 2.0))

        free = 19_000 * estimate.MiB
        est = estimate.estimate(info, n_ctx=32768, device_free_bytes=free,
                                ubatch=512)
        expected_kv = 32 * 32768 * 8 * 256 * 2
        self.assertEqual(est.kv_bytes, expected_kv)
        self.assertAlmostEqual(est.kv_bytes / estimate.GiB, 4.0, places=2)
        # Weights are the raw tensor bytes plus the architecture overhead.
        self.assertGreater(est.weights_bytes, info.total_tensor_bytes)
        self.assertLess(est.weights_bytes, info.total_tensor_bytes * 1.10)
        self.assertTrue(est.fits)
        self.assertGreater(est.headroom_bytes, 0)

        # Halving the available VRAM must not increase the suggested context.
        half = estimate.estimate(info, n_ctx=32768, device_free_bytes=free // 2,
                                 ubatch=512)
        self.assertLessEqual(half.max_ctx_fit, est.max_ctx_fit)

    def test_max_context_ladder_picks_a_sane_value(self) -> None:
        """With tight VRAM the suggestion must be a real ladder value, not 0."""
        path = build_gguf(
            self.dir / "tight.gguf",
            typical_metadata(n_layer=32, n_head=32, n_head_kv=8, n_embd=4096,
                             n_ctx_train=131072),
            [("token_embd.weight", (4096, 152064), 12)],
        )
        info = gguf.load(str(path))
        # Room for roughly 2 GiB of KV => ~16k tokens at 131072 B/token.
        suggestion = estimate.suggest_context(info, device_free_bytes=6200 * estimate.MiB)
        self.assertIn(suggestion, estimate.CTX_CANDIDATES + (0,))
        self.assertGreater(suggestion, 0)
        self.assertLess(suggestion, 131072)

        # And that suggestion must actually fit when re-estimated.
        est = estimate.estimate(info, n_ctx=suggestion, device_free_bytes=6200 * estimate.MiB)
        self.assertTrue(est.fits, f"建议上下文 {suggestion} 自相矛盾：估算并不 fit")

    def test_suggest_context_is_monotonic_in_vram(self) -> None:
        info = self._info()
        low = estimate.suggest_context(info, device_free_bytes=6 * estimate.GiB)
        high = estimate.suggest_context(info, device_free_bytes=18 * estimate.GiB)
        self.assertGreaterEqual(high, low)

    def test_human_bytes(self) -> None:
        self.assertEqual(estimate.human_bytes(0), "0 B")
        self.assertEqual(estimate.human_bytes(1024), "1.00 KiB")
        self.assertEqual(estimate.human_bytes(2 * estimate.GiB), "2.00 GiB")
        self.assertTrue(estimate.human_bytes(-estimate.GiB).startswith("-"))

    def test_broken_header_yields_warning_not_exception(self) -> None:
        path = self.dir / "bad.gguf"
        path.write_bytes(b"XXXX" + b"\x00" * 32)
        info = gguf.load(str(path))
        est = estimate.estimate(info, n_ctx=4096, device_free_bytes=8 * estimate.GiB)
        self.assertTrue(est.warnings)
        self.assertEqual(est.kv_bytes, 0)


# -------------------------------------------------------------------- argspec


class ArgSpecTests(unittest.TestCase):
    def test_table_is_non_trivial_and_consistent(self) -> None:
        self.assertGreater(len(argspec.SPECS), 150)
        for flag, spec in argspec.SPECS.items():
            with self.subTest(flag=flag):
                self.assertEqual(flag, spec.flag)
                self.assertTrue(flag.startswith("--"))
                self.assertIsInstance(spec.label, str)
                self.assertTrue(spec.label)

    def test_no_duplicate_flags(self) -> None:
        flags = [spec.flag for spec in argspec._specs()]
        self.assertEqual(len(flags), len(set(flags)), "参数表存在重复项")

    def test_bool_serialisation_uses_negation(self) -> None:
        spec = argspec.get("--kv-offload")
        self.assertEqual(spec.serialize(True), ["--kv-offload"])
        self.assertEqual(spec.serialize(False), ["--no-kv-offload"])
        self.assertEqual(spec.serialize(None), [])

    def test_tristate_always_carries_its_value(self) -> None:
        spec = argspec.get("--flash-attn")
        self.assertEqual(spec.serialize("on"), ["--flash-attn", "on"])
        self.assertEqual(spec.serialize(""), [])

    def test_choice_rejects_unknown_values(self) -> None:
        spec = argspec.get("--rope-scaling")
        with self.assertRaises(ValueError):
            spec.coerce("bogus")
        self.assertEqual(spec.coerce("yarn"), "yarn")

    def test_int_range_enforced(self) -> None:
        spec = argspec.get("--port")
        self.assertEqual(spec.coerce("8081"), 8081)
        with self.assertRaises(ValueError):
            spec.coerce("0")
        with self.assertRaises(ValueError):
            spec.coerce("70000")
        with self.assertRaises(ValueError):
            spec.coerce("abc")

    def test_float_serialisation_avoids_trailing_noise(self) -> None:
        spec = argspec.get("--temp")
        self.assertEqual(spec.serialize(0.7), ["--temp", "0.7"])
        self.assertEqual(spec.serialize(1), ["--temp", "1"])

    def test_repeat_serialisation(self) -> None:
        spec = argspec.get("--lora")
        self.assertEqual(spec.serialize(["a.gguf"]), ["--lora", "a.gguf"])
        self.assertEqual(
            spec.serialize(["a.gguf", "b.gguf"]),
            ["--lora", "a.gguf", "--lora", "b.gguf"],
        )

    def test_path_with_spaces_is_a_single_token(self) -> None:
        spec = argspec.get("--model")
        tokens = spec.serialize(r"C:\My Models\qwen 7b.gguf")
        self.assertEqual(tokens, ["--model", r"C:\My Models\qwen 7b.gguf"])
        self.assertEqual(len(tokens), 2)

    def test_is_default(self) -> None:
        self.assertTrue(argspec.get("--temp").is_default(0.8))
        self.assertTrue(argspec.get("--temp").is_default("0.80"))
        self.assertFalse(argspec.get("--temp").is_default(0.9))
        self.assertTrue(argspec.get("--metrics").is_default(False))

    def test_canonical_maps_negative_forms(self) -> None:
        self.assertEqual(argspec.canonical("--no-kv-offload"), "--kv-offload")
        self.assertEqual(argspec.canonical("--no-webui"), "--ui")
        self.assertEqual(argspec.canonical("--unknown-flag"), "--unknown-flag")

    def test_groups_cover_the_table(self) -> None:
        covered = set()
        for group in argspec.Group:
            covered.update(spec.flag for spec in argspec.specs_for(group))
        self.assertEqual(covered, set(argspec.SPECS))

    def test_danger_flags_are_marked(self) -> None:
        for flag in ("--tools", "--agent", "--ui-mcp-proxy", "--mcp-servers-config"):
            with self.subTest(flag=flag):
                self.assertTrue(argspec.get(flag).danger)


# -------------------------------------------------------------------- config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        from llamagui.paths import Layout

        self.store = config_mod.Store(
            Layout(
                root=self.base,
                gui_dir=self.base / "gui",
                config_dir=self.base / "cfg",
                presets_dir=self.base / "cfg" / "presets",
                cache_dir=self.base / "cfg" / "cache",
                logs_dir=self.base / "cfg" / "logs",
                bin_dir=self.base,
                help_override=None,
                source="test",
            )
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_round_trip(self) -> None:
        cfg = config_mod.Config()
        cfg.set("--model", r"C:\models\a.gguf")
        cfg.set("--ctx-size", 8192)
        cfg.set("--flash-attn", "on")
        cfg.env["LLAMA_CACHE"] = r"D:\cache"
        cfg.push_recent("C:/x.gguf")
        self.store.save(cfg)

        loaded = self.store.load()
        self.assertEqual(loaded.get("--model"), r"C:\models\a.gguf")
        self.assertEqual(loaded.get("--ctx-size"), 8192)
        self.assertEqual(loaded.get("--flash-attn"), "on")
        self.assertEqual(loaded.env["LLAMA_CACHE"], r"D:\cache")
        self.assertEqual(loaded.recent_models, ["C:/x.gguf"])

    def test_none_values_survive_the_round_trip(self) -> None:
        cfg = config_mod.Config()
        cfg.set("--ctx-size", None)
        self.store.save(cfg)
        loaded = self.store.load()
        self.assertIn("--ctx-size", loaded.values)
        self.assertIsNone(loaded.values["--ctx-size"])
        self.assertNotIn("--ctx-size", loaded.explicit_flags)

    def test_missing_file_returns_defaults(self) -> None:
        loaded = self.store.load()
        self.assertEqual(loaded.values, {})
        self.assertEqual(loaded.recent_models, [])

    def test_corrupt_file_is_quarantined(self) -> None:
        self.store.layout.config_dir.mkdir(parents=True, exist_ok=True)
        (self.store.layout.config_dir / "config.json").write_text("{not json", encoding="utf-8")
        loaded = self.store.load()
        self.assertEqual(loaded.values, {})
        self.assertTrue((self.store.layout.config_dir / "config.json.corrupt").is_file())

    def test_presets_crud(self) -> None:
        preset = config_mod.Preset(name="Qwen 代码", values={"--temp": 0.2},
                                   description="低温度", model="qwen.gguf")
        path = self.store.save_preset(preset)
        self.assertTrue(path.is_file())

        found = self.store.list_presets()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "Qwen 代码")
        self.assertEqual(found[0].values["--temp"], 0.2)

        loaded = self.store.load_preset("Qwen 代码")
        self.assertIsNotNone(loaded)
        self.assertGreater(loaded.created, 0)

        renamed = self.store.rename_preset("Qwen 代码", "Qwen 编码")
        self.assertIsNotNone(renamed)
        self.assertIsNone(self.store.load_preset("Qwen 代码"))
        self.assertIsNotNone(self.store.load_preset("Qwen 编码"))

        self.assertTrue(self.store.delete_preset("Qwen 编码"))
        self.assertFalse(self.store.delete_preset("Qwen 编码"))

    def test_preset_name_sanitisation(self) -> None:
        self.assertEqual(config_mod.sanitize_name("a/b:c*d"), "a_b_c_d")
        self.assertEqual(config_mod.sanitize_name("   "), "preset")
        self.assertEqual(config_mod.sanitize_name("..."), "preset")

    def test_recent_models_deduplicates_and_caps(self) -> None:
        cfg = config_mod.Config()
        for i in range(20):
            cfg.push_recent(f"m{i}.gguf", limit=5)
        self.assertEqual(len(cfg.recent_models), 5)
        self.assertEqual(cfg.recent_models[0], "m19.gguf")
        cfg.push_recent("m17.gguf", limit=5)
        self.assertEqual(cfg.recent_models[0], "m17.gguf")
        self.assertEqual(len(cfg.recent_models), 5)

    def test_merge_values_skips_none(self) -> None:
        base = {"--temp": 0.7, "--top-k": 40}
        merged = config_mod.merge_values(base, {"--temp": None, "--top-k": 20})
        self.assertEqual(merged, {"--temp": 0.7, "--top-k": 20})

    def test_merge_values_restricts_keys(self) -> None:
        merged = config_mod.merge_values({"--a": 1}, {"--a": 2, "--b": 3}, keys=["--b"])
        self.assertEqual(merged, {"--a": 1, "--b": 3})

    def test_cache_helpers(self) -> None:
        self.assertIsNone(self.store.read_cache("nope.txt"))
        self.store.write_cache("x.txt", "hello")
        self.assertEqual(self.store.read_cache("x.txt"), "hello")
        self.store.write_cache_json("d.json", {"a": 1})
        self.assertEqual(self.store.read_cache_json("d.json"), {"a": 1})

    def test_log_path_is_timestamped(self) -> None:
        path = self.store.log_path("launch")
        self.assertEqual(path.suffix, ".txt")
        self.assertIn("launch-", path.name)


# ------------------------------------------------------------------- cmdline


@unittest.skipUnless(HELP_SNAPSHOT.is_file(), "help snapshot not present")
class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = helpparse.parse_help(
            HELP_SNAPSHOT.read_text(encoding="utf-8", errors="replace"), "llama-server"
        )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        # Relative names used throughout the tests resolve to this real file.
        (self.base / "a.gguf").write_bytes(b"GGUF")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, values: dict, **kwargs):
        cfg = config_mod.Config(values=dict(values))
        # A relative model path is resolved against the directory the server is
        # started in, exactly as the launcher does; point that at a temp dir so
        # validation sees a real file.
        kwargs.setdefault("base_dir", self.base)
        return cmdline.build(cfg, help_doc=self.doc, **kwargs)

    def _real_model(self, name: str = "a.gguf") -> str:
        """Create a real (empty) .gguf so path validation succeeds."""
        path = self.base / name
        path.write_bytes(b"GGUF")
        return str(path)

    def test_minimal_model_command(self) -> None:
        result = self._build({"--model": self._real_model()})
        self.assertEqual(len(result.argv), 2)
        self.assertEqual(result.argv[0], "--model")
        self.assertTrue(result.argv[1].endswith("a.gguf"))
        self.assertTrue(result.validation.ok, result.validation.errors)

    def test_unset_values_are_not_emitted(self) -> None:
        # This is the property that keeps --fit usable: a GUI default that the
        # user never touched must not appear on the command line.
        result = self._build({
            "--model": self._real_model(),
            "--ctx-size": None,
            "--temp": None,
        })
        self.assertEqual(result.argv[0], "--model")
        self.assertNotIn("--ctx-size", result.argv)
        self.assertNotIn("--temp", result.argv)
        self.assertEqual(len(result.argv), 2)

    def test_explicit_default_is_emitted(self) -> None:
        result = self._build({"--model": self._real_model(), "--fit": "off"})
        self.assertIn("--fit", result.argv)
        self.assertIn("off", result.argv)

    def test_flag_order_follows_the_table(self) -> None:
        result = self._build({
            "--model": self._real_model(),
            "--temp": 0.5,
            "--ctx-size": 4096,
        })
        argv = result.argv
        self.assertLess(argv.index("--model"), argv.index("--ctx-size"))
        self.assertLess(argv.index("--ctx-size"), argv.index("--temp"))

    def test_bool_negation_is_emitted(self) -> None:
        result = self._build({"--model": self._real_model(), "--kv-offload": False})
        self.assertIn("--no-kv-offload", result.argv)
        self.assertNotIn("--kv-offload", result.argv)

    def test_unknown_flag_is_reported_not_emitted(self) -> None:
        result = self._build({"--model": self._real_model(), "--totally-made-up": "x"})
        self.assertIn("--totally-made-up", result.unknown)
        self.assertNotIn("--totally-made-up", result.argv)

    def test_documented_but_uncurated_flag_is_forwarded(self) -> None:
        # Present in this build's help but not in the curated table: forward it
        # verbatim so presets written for a newer build still work.
        result = self._build({"--model": self._real_model(), "--numa": "distribute"})
        self.assertIn("--numa", result.argv)
        self.assertNotIn("--numa", result.unknown)

    def test_negated_flag_keeps_correct_polarity(self) -> None:
        # --no-host is itself a negation.  It must only ever appear when the
        # user ticks it; an untouched checkbox must not disable the host buffer
        # as a side effect of "False means no host buffer".
        spec = argspec.get("--no-host")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.neg, "")
        self.assertEqual(spec.serialize(True), ["--no-host"])
        self.assertEqual(spec.serialize(False), [])
        self.assertEqual(spec.serialize(None), [])
        # Unticked (False) must never emit the flag.
        self.assertFalse(spec.is_default(True))
        self.assertTrue(spec.is_default(False))

        result = self._build({"--model": self._real_model()})
        self.assertNotIn("--no-host", result.argv)

        result = self._build({"--model": self._real_model(), "--no-host": True})
        self.assertEqual(result.argv.count("--no-host"), 1)

    def test_deprecated_argument_is_rejected_by_the_help_check(self) -> None:
        # --draft was removed in favour of --spec-draft-n-max.  It must not be
        # forwarded blindly; it should surface as unknown for this build.
        result = self._build({"--model": self._real_model(), "--draft": 8})
        self.assertIn("--draft", result.unknown)
        self.assertNotIn("--draft", result.argv)

    def test_missing_model_is_an_error(self) -> None:
        result = self._build({"--temp": 0.5})
        self.assertFalse(result.validation.ok)
        self.assertTrue(any("未指定模型" in e for e in result.validation.errors))

    def test_router_mode_needs_no_model(self) -> None:
        result = self._build({"--models-dir": "."})
        self.assertTrue(result.validation.ok)
        self.assertTrue(result.router_mode)

    def test_router_mode_conflicts_with_model(self) -> None:
        result = self._build({"--model": self._real_model(), "--models-dir": "."})
        self.assertFalse(result.validation.ok)
        self.assertTrue(any("路由模式" in e for e in result.validation.errors))

    def test_non_unified_kv_with_parallel_warns(self) -> None:
        result = self._build({
            "--model": "a.gguf", "--parallel": 4, "--kv-unified": False,
            "--ctx-size": 8192,
        })
        self.assertTrue(any("KV" in w for w in result.validation.warnings))

    def test_parallel_with_unified_kv_informs_about_slot_context(self) -> None:
        result = self._build({"--model": self._real_model(), "--parallel": 4, "--ctx-size": 8192})
        self.assertTrue(any("单槽" in i for i in result.validation.infos))

    def test_ubatch_over_batch_is_an_error(self) -> None:
        result = self._build({"--model": self._real_model(), "--batch-size": 256, "--ubatch-size": 512})
        self.assertFalse(result.validation.ok)

    def test_ubatch_not_multiple_of_32_warns(self) -> None:
        result = self._build({"--model": self._real_model(), "--ubatch-size": 100})
        self.assertTrue(any("32" in w for w in result.validation.warnings))

    def test_ubatch_over_context_is_an_error(self) -> None:
        result = self._build({"--model": self._real_model(), "--ctx-size": 256, "--ubatch-size": 512})
        self.assertFalse(result.validation.ok)

    def test_exposed_host_with_tools_is_refused(self) -> None:
        result = self._build({"--model": self._real_model(), "--host": "0.0.0.0", "--tools": "all"})
        self.assertFalse(result.validation.ok)
        self.assertTrue(any("拒绝启动" in e for e in result.validation.errors))

    def test_exposed_host_alone_only_warns(self) -> None:
        result = self._build({"--model": self._real_model(), "--host": "0.0.0.0"})
        self.assertTrue(result.validation.ok)
        self.assertTrue(any("暴露" in w for w in result.validation.warnings))

    def test_missing_model_file_is_an_error(self) -> None:
        result = self._build({"--model": r"C:\definitely\not\here.gguf"})
        self.assertFalse(result.validation.ok)
        self.assertTrue(any("不存在" in e for e in result.validation.errors))

    def test_non_gguf_extension_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            path.write_bytes(b"x")
            result = self._build({"--model": str(path)})
        self.assertTrue(result.validation.ok)
        self.assertTrue(any(".gguf" in w for w in result.validation.warnings))

    def test_draft_speculation_without_model_warns(self) -> None:
        result = self._build({"--model": self._real_model(), "--spec-type": "draft-simple"})
        self.assertTrue(any("草稿" in w for w in result.validation.warnings))

    def test_ngram_speculation_does_not_require_a_draft_model(self) -> None:
        result = self._build({"--model": self._real_model(), "--spec-type": "ngram-mod"})
        self.assertTrue(result.validation.ok)

    def test_invalid_choice_is_reported_as_error(self) -> None:
        result = self._build({"--model": self._real_model(), "--flash-attn": "maybe"})
        self.assertFalse(result.validation.ok)

    def test_env_overrides_are_collected(self) -> None:
        cfg = config_mod.Config(values={"--model": "a.gguf"})
        cfg.env = {"LLAMA_CACHE": r"D:\hf", "": "ignored"}
        result = cmdline.build(cfg, help_doc=self.doc)
        self.assertEqual(result.env, {"LLAMA_CACHE": r"D:\hf"})

    def test_quoting_for_preview_only(self) -> None:
        self.assertEqual(cmdline.quote_token("plain"), "plain")
        self.assertEqual(cmdline.quote_token(r"C:\My Models\a.gguf"), r'"C:\My Models\a.gguf"')
        result = self._build({"--model": r"C:\My Models\a b.gguf"})
        # argv keeps the path intact as one token; only the preview quotes it.
        self.assertEqual(result.argv[1], r"C:\My Models\a b.gguf")
        self.assertIn('"C:\\My Models\\a b.gguf"', result.preview())

    def test_multiline_preview_wraps(self) -> None:
        result = self._build({"--model": self._real_model(), "--ctx-size": 8192})
        text = result.multiline(width=40)
        self.assertIn("\n", text)

    def test_port_available_and_suggestion(self) -> None:
        import socket

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        try:
            # A genuinely listening socket must be reported as occupied...
            self.assertFalse(cmdline.port_available("127.0.0.1", port))
            # ...and the suggestion must not be that port, and must be one we
            # can actually bind.
            suggestion = cmdline.suggest_port("127.0.0.1", port)
            self.assertNotEqual(suggestion, port)
            self.assertTrue(cmdline.port_available("127.0.0.1", suggestion))
        finally:
            listener.close()

    def test_bound_but_unlistening_socket_is_occupied(self) -> None:
        # On Windows such a socket silently drops SYNs, so a connect probe would
        # have to time out to notice.  The bind-based probe reports it correctly
        # and immediately.
        import socket

        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        try:
            self.assertFalse(cmdline.port_available("127.0.0.1", port))
        finally:
            holder.close()

    def test_port_probe_is_fast(self) -> None:
        # suggest_port probes a range; a connect-based probe would cost a full
        # timeout per candidate and make the UI hang.
        import time

        started = time.perf_counter()
        for port in range(20000, 20060):
            cmdline.port_available("127.0.0.1", port)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0, f"60 次端口探测耗时 {elapsed:.2f}s，过慢")

    def test_invalid_ports_are_not_available(self) -> None:
        for bad in (0, -1, 70000, None, "abc"):
            with self.subTest(port=bad):
                self.assertFalse(cmdline.port_available("127.0.0.1", bad))

    def test_removed_argument_is_never_forwarded(self) -> None:
        # --draft is documented as removed; the parser must notice and the
        # builder must refuse it rather than letting llama-server abort.
        removed = [f for f, opt in self.doc.options.items() if opt.removed]
        self.assertIn("--draft", removed)
        result = self._build({"--model": self._real_model(), "--draft": 8})
        self.assertIn("--draft", result.unknown)
        self.assertNotIn("--draft", result.argv)

    def test_managed_flags_are_never_forwarded(self) -> None:
        result = self._build({"--model": self._real_model(), "--help": True, "--version": True})
        self.assertNotIn("--help", result.argv)
        self.assertNotIn("--version", result.argv)


# ------------------------------------------------------------------- logparse


class LogParseTests(unittest.TestCase):
    def test_strip_ansi(self) -> None:
        self.assertEqual(logparse.strip_ansi("\x1b[31merror\x1b[0m"), "error")
        self.assertEqual(logparse.strip_ansi("plain"), "plain")
        self.assertEqual(logparse.strip_ansi("\x1b]0;title\x07rest"), "rest")

    def test_level_detection(self) -> None:
        self.assertEqual(logparse.guess_level("E llama_context: error"), "error")
        self.assertEqual(logparse.guess_level("W llama: warning"), "warn")
        self.assertEqual(logparse.guess_level("I main: info"), "info")
        self.assertEqual(logparse.guess_level("something failed badly"), "error")
        self.assertEqual(logparse.guess_level("nothing notable"), "info")

    def test_extracts_context_and_kv_size(self) -> None:
        parser = logparse.LogParser()
        parser.parse_line("llama_context: n_ctx         = 8192")
        parser.parse_line("llama_context: n_batch       = 2048")
        parser.parse_line("llama_context: n_ubatch      = 512")
        parser.parse_line("llama_kv_cache: KV self size = 1024.00 MiB, "
                          "K (f16): 512.00 MiB, V (f16): 512.00 MiB")
        parser.parse_line("load_tensors: offloaded 33/33 layers to GPU")
        f = parser.facts
        self.assertEqual(f.n_ctx, 8192)
        self.assertEqual(f.n_batch, 2048)
        self.assertEqual(f.n_ubatch, 512)
        self.assertEqual(f.kv_size_mib, 1024.0)
        self.assertEqual(f.kv_k_type, "f16")
        self.assertEqual(f.offloaded_layers, 33)
        self.assertEqual(f.total_layers, 33)

    def test_extracts_throughput(self) -> None:
        parser = logparse.LogParser()
        parser.parse_line("prompt eval time =   123.45 ms /    64 tokens "
                          "(   1.92 ms per token,   518.42 tokens per second)")
        parser.parse_line("        eval time =  1000.00 ms /   100 tokens "
                          "(  10.00 ms per token,   100.00 tokens per second)")
        self.assertEqual(parser.facts.prompt_per_second, 518.42)
        self.assertEqual(parser.facts.eval_per_second, 100.0)
        self.assertIn("100.0 t/s", parser.facts.summary())

    def test_extracts_devices_and_backends(self) -> None:
        parser = logparse.LogParser()
        parser.parse_line("ggml_cuda_init: found 1 CUDA devices (Total VRAM: 20479 MiB):")
        parser.parse_line("  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6")
        parser.parse_line("load_backend: loaded CUDA backend from F:\\llama.cpp\\ggml-cuda.dll")
        parser.parse_line("load_backend: loaded CPU backend from F:\\llama.cpp\\ggml-cpu.dll")
        parser.parse_line("  CUDA0: NVIDIA GeForce RTX 3080 (20479 MiB, 19297 MiB free)")
        self.assertIn("CUDA", parser.facts.backends)
        self.assertIn("CPU", parser.facts.backends)
        self.assertEqual(len(parser.facts.devices), 1)
        self.assertIn("RTX 3080", parser.facts.devices[0])
        # The init banner's "Device 0:" summary line must not be mistaken for a
        # device listing.
        self.assertFalse(any("compute capability" in d for d in parser.facts.devices))

    def test_buffer_sizes_accumulate(self) -> None:
        parser = logparse.LogParser()
        parser.parse_line("CUDA0 model buffer size = 4096.00 MiB")
        parser.parse_line("CUDA0 KV buffer size = 1024.00 MiB")
        parser.parse_line("CUDA0 compute buffer size = 512.00 MiB")
        self.assertAlmostEqual(parser.facts.buffers_mib["CUDA0 model"], 4096.0)
        self.assertAlmostEqual(parser.facts.buffers_total_mib, 5632.0)

    def test_jsonl_records_are_parsed(self) -> None:
        parser = logparse.LogParser()
        event = parser.parse_line(json.dumps({
            "level": "error",
            "msg": "failed to allocate",
            "func": "llama_model_load",
            "n_ctx": 4096,
        }))
        self.assertIsNotNone(event)
        self.assertEqual(event.level, "error")
        self.assertIn("failed to allocate", event.text)
        self.assertIn("llama_model_load", event.text)
        self.assertEqual(parser.facts.n_ctx, 4096)

    def test_jsonl_warning_level(self) -> None:
        parser = logparse.LogParser()
        event = parser.parse_line(json.dumps({"level": "warn", "msg": "buffer reuse"}))
        self.assertEqual(event.level, "warn")

    def test_malformed_json_falls_back_to_text(self) -> None:
        parser = logparse.LogParser()
        event = parser.parse_line('{"level": "info", broken')
        self.assertIsNotNone(event)
        self.assertEqual(event.level, "info")

    def test_blank_lines_are_ignored(self) -> None:
        parser = logparse.LogParser()
        self.assertIsNone(parser.parse_line(""))
        self.assertIsNone(parser.parse_line("   \n"))
        self.assertEqual(parser.line_count, 0)

    def test_milestones(self) -> None:
        parser = logparse.LogParser()
        self.assertEqual(
            parser.parse_line("main: server is listening on 127.0.0.1:8080").milestone,
            "listening",
        )
        self.assertEqual(
            parser.parse_line("llama_model_loader: failed to load model").milestone,
            "failed",
        )

    def test_counters(self) -> None:
        parser = logparse.LogParser()
        parser.parse_line("E something error")
        parser.parse_line("W something warn")
        parser.parse_line("I fine")
        self.assertEqual(parser.error_count, 1)
        self.assertEqual(parser.warning_count, 1)
        self.assertEqual(parser.line_count, 3)

    def test_reset_clears_everything(self) -> None:
        parser = logparse.LogParser()
        parser.parse_line("llama_context: n_ctx = 4096")
        parser.reset()
        self.assertIsNone(parser.facts.n_ctx)
        self.assertEqual(parser.line_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Parse ``--help`` output from the local llama.cpp binaries.

The installed build is the authority on which arguments exist and what values
they accept.  This module turns a help dump into structured
:class:`OptionDef` records so the UI can grey out arguments a particular build
does not support, and so the curated metadata table in :mod:`argspec` can be
cross-checked instead of trusted blindly.

The parser is defensive by design: help formatting has changed several times
upstream, so anything it cannot understand is recorded in
:attr:`HelpDoc.unparsed` rather than raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Long-form argument, e.g. ``--ctx-size`` / ``--no-kv-offload``.
_LONG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9._-]*")
#: Short-form argument, e.g. ``-c`` / ``-ngl`` / ``-no-kvu`` / ``-no-ag``.
#: llama.cpp documents internal negated toggles as hyphenated short forms with a
#: single dash, so the body may contain hyphens; only the leading dash is the
#: sigil.  The first character must be a letter so a negative number in a
#: description is never mistaken for a flag, and the lookahead requires the
#: delimiter help text actually uses between flags (``,`` or alignment spaces),
#: which keeps prose such as ``--logit-bias EOS-inf`` from matching.
_SHORT_RE = re.compile(r"(-[A-Za-z][A-Za-z0-9.-]*)(?=\s*(?:,|\s{2,}|$))")
#: ``(env: SOME_NAME)`` or ``(env: MTMD_BACKEND_DEVICE)``.  The name group is
#: deliberately permissive: one upstream entry reads ``(env: LLAMA_ARG_TOOLS_R``
#: because the generated docs truncate it.
_ENV_RE = re.compile(r"\(env:\s*([A-Za-z_][A-Za-z0-9_]*)[^)]*\)")
#: ``allowed values: a, b, c``
_ALLOWED_RE = re.compile(r"allowed values:\s*(.+)", re.IGNORECASE)
#: ``(default: X)`` / ``(default: X, ...)`` - capture up to the first comma that
#: is not inside brackets, or the end of the parenthesis.
_DEFAULT_RE = re.compile(r"\(default:\s*([^)]*)", re.IGNORECASE)
#: Placeholders such as ``{none,linear,yarn}`` / ``<0|1>`` / ``[on|off|auto]``.
_BRACE_RE = re.compile(r"\{([^{}]+)\}")
_ANGLE_RE = re.compile(r"<([^<>]+)>")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
#: A section separator line, e.g. ``----- common params -----``.
_SECTION_RE = re.compile(r"^-{3,}\s*(.*?)\s*-{3,}\s*$")
#: Argument-taking placeholders that are not enum-like.
_PLACEHOLDER_RE = re.compile(r"^[A-Z][A-Z0-9_.,+:/|<>\[\]-]*$")
#: Alternatives written ``{a,b}`` / ``<a|b>`` / ``[a|b]``.
_WRAP_RE = re.compile(r"<[^<>]*>|\[[^\[\]]*\]|\{[^{}]*\}")
#: A bare placeholder word such as ``N``, ``TYPE``, ``SCHEMA``, ``lo-hi``.
_ARGWORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.,+:/|_=-]*")
#: Token separator inside the usage region.
_SEP_RE = re.compile(r"[\s,]+")
#: Column at which llama.cpp aligns option descriptions.  Descriptions always
#: start at or after this column, which is what makes header splitting
#: unambiguous even for options whose usage text is long.
_DESC_COLUMN = 40
#: Below this width there is no room for a description at all.
_MIN_DESC_GAP = 2


@dataclass
class OptionDef:
    """One command line option as described by the binary's own help."""

    long: str
    shorts: Tuple[str, ...] = ()
    #: Additional long forms documented for the same option, e.g.
    #: ``--n-predict`` for ``--predict`` or ``--no-kv-offload`` for
    #: ``--kv-offload``.
    aliases: Tuple[str, ...] = ()
    placeholders: Tuple[str, ...] = ()
    #: Values harvested from ``allowed values:`` or from brace/angle/bracket
    #: enums in the usage string.
    choices: Tuple[str, ...] = ()
    #: Raw default text, e.g. ``0.80``, ``-1, -1 = infinity``, ``auto``.
    default_text: Optional[str] = None
    env: Optional[str] = None
    help_text: str = ""
    #: Text on the defining line, after the usage region.  Defaults and
    #: ``allowed values`` are read only from here, never from continuation
    #: lines, because a following option's ``allowed values:`` block would
    #: otherwise be attributed to this one.
    own_text: str = ""
    section: str = ""
    line_number: int = 0

    @property
    def takes_value(self) -> bool:
        """True when the option expects an argument."""
        if self.placeholders:
            return True
        # ``--fit [on|off]`` style: optional-looking but llama.cpp requires it.
        return bool(self.choices) and self.default_text is not None

    @property
    def allows_no_value(self) -> bool:
        """True when the option is documented as a bare toggle."""
        return not self.placeholders and not self.choices

    @property
    def removed(self) -> bool:
        """True when the help text marks this option as removed or deprecated.

        llama.cpp keeps removed arguments in ``--help`` so that stale command
        lines fail with a pointer to the replacement instead of a bare
        "unknown argument".  Forwarding such an option verbatim would abort the
        launch, so callers must treat it as unsupported.
        """
        text = self.help_text.lower()
        return (
            "has been removed" in text
            or "argument has been removed" in text
            or "removed. use" in text
        )

    def default_parsed(self) -> object:
        """Best-effort conversion of :attr:`default_text` to a Python value.

        Returns ``None`` when there is no default or it is not decomposable.
        """
        if not self.default_text:
            return None
        text = self.default_text.strip()
        # Drop trailing qualifiers such as ", same as --threads" or
        # ", 0 = disabled, 1 = Mirostat".
        head = text.split(",")[0].strip()
        if head in ("", "unused"):
            return None
        if head.lower() == "true":
            return True
        if head.lower() == "false":
            return False
        try:
            if any(ch in head for ch in ".eE") and not head.startswith("0x"):
                return float(head)
            return int(head)
        except ValueError:
            return head

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.long


@dataclass
class HelpDoc:
    """A parsed ``--help`` dump."""

    source: str
    build_info: str = ""
    options: Dict[str, OptionDef] = field(default_factory=dict)
    #: Lines that looked like option definitions but could not be parsed.
    unparsed: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)

    def get(self, long_name: str) -> Optional[OptionDef]:
        return self.options.get(long_name)

    def has(self, long_name: str) -> bool:
        return long_name in self.options

    def missing(self, long_names) -> List[str]:  # noqa: ANN001
        """Which of *long_names* this build does not document."""
        return [n for n in long_names if n not in self.options]


# --------------------------------------------------------------- line handling


def split_usage_line(line: str) -> Tuple[str, str]:
    """Split a help line into ``(usage_region, description)``.

    llama.cpp aligns every option description at a fixed column, and the usage
    region before it consists only of flags, commas, and value placeholders.  A
    description therefore begins at the first run of two or more spaces that
    ends at or after :data:`_DESC_COLUMN`.  Options whose usage text is long
    enough to reach that column (``--override-tensor <pattern>=<buffer>,...``)
    simply have no description on their defining line, which is exactly right:
    their description is indented on the following line.
    """
    stripped = line.rstrip()
    search_from = 0
    while True:
        gap = stripped.find("  ", search_from)
        if gap == -1:
            return stripped, ""
        end = gap
        while end < len(stripped) and stripped[end] == " ":
            end += 1
        if end - gap >= _MIN_DESC_GAP and end >= _DESC_COLUMN and end < len(stripped):
            return stripped[:gap].rstrip(), stripped[end:].strip()
        # A wide gap that is not the description column (e.g. ``-t,    --threads``
        # inside the usage region) - keep scanning.
        search_from = end if end > gap else gap + 2


def _scan_usage(usage: str) -> Tuple[List[str], List[str]]:
    """Walk *usage* left to right into ``(flags, placeholder_args)``.

    Anything that is neither a flag nor a plausible placeholder stops the scan;
    the caller ignores whatever is left because the description has already been
    removed by :func:`split_usage_line`.
    """
    flags: List[str] = []
    args: List[str] = []
    pos = 0
    n = len(usage)
    while pos < n:
        m = _SEP_RE.match(usage, pos)
        if m:
            pos = m.end()
            continue
        if usage.startswith("--", pos):
            m = _LONG_RE.match(usage, pos)
            if m:
                flags.append(m.group(0))
                pos = m.end()
                continue
        if usage[pos] == "-" and pos + 1 < n and usage[pos + 1] != "-":
            m = _SHORT_RE.match(usage, pos)
            if m:
                flags.append(m.group(1))
                pos = m.end()
                continue
        m = _WRAP_RE.match(usage, pos)
        if m:
            args.append(m.group(0))
            pos = m.end()
            continue
        m = _ARGWORD_RE.match(usage, pos)
        if m:
            args.append(m.group(0))
            pos = m.end()
            continue
        break
    return flags, args


def _split_usage(usage: str) -> Tuple[List[str], List[str], str]:
    """Split the usage region into ``(longs, shorts, placeholder remainder)``."""
    flags, args = _scan_usage(usage)
    longs = [f for f in flags if f.startswith("--")]
    shorts = [f for f in flags if not f.startswith("--")]
    return longs, shorts, " ".join(args).strip()


def _usable_choices(text: str) -> Tuple[str, ...]:
    """Extract enum-like values from a placeholder or ``allowed values`` list."""
    text = text.strip().rstrip(".")
    # ``none,draft-simple,...`` and ``f32, f16, bf16`` are both plausible.
    parts = [p.strip().strip("`'\"") for p in text.split(",")]
    parts = [p for p in parts if p and _is_plain_value(p)]
    if len(parts) >= 2:
        return tuple(dict.fromkeys(parts))
    return ()


def _is_plain_value(token: str) -> bool:
    if not token or len(token) > 24:
        return False
    return all(ch.isalnum() or ch in "-_+." for ch in token)


def _choices_from_usage(remainder: str) -> Tuple[str, ...]:
    """Pull ``{a,b}`` / ``<a|b>`` / ``[a|b]`` alternatives out of a usage string."""
    for match in _BRACE_RE.findall(remainder):
        vals = _usable_choices(match)
        if vals:
            return vals
    for pattern in (_ANGLE_RE, _BRACKET_RE):
        for match in pattern.findall(remainder):
            if "|" in match:
                vals = tuple(v.strip() for v in match.split("|") if v.strip())
                if vals:
                    return vals
    return ()


def _is_option_start(line: str) -> bool:
    """True when *line* defines one or more options (rather than continuing one)."""
    stripped = line.strip()
    if not stripped.startswith("-"):
        return False
    if _SECTION_RE.match(stripped):
        return False
    usage, _desc = split_usage_line(line)
    return bool(_LONG_RE.search(usage))


def parse_help(text: str, source: str = "") -> HelpDoc:
    """Parse help *text* into a :class:`HelpDoc`."""
    doc = HelpDoc(source=source)
    lines = text.splitlines()

    # Build-version banner, if present (llama.cpp prints it before the options).
    for line in lines[:8]:
        if line.lower().startswith("version:"):
            doc.build_info = line.split(":", 1)[1].strip()
            break

    section = ""
    current: Optional[OptionDef] = None
    desc_lines: List[str] = []

    def flush() -> None:
        nonlocal current, desc_lines
        if current is not None:
            body = "\n".join(desc_lines).strip()
            current.help_text = body
            current.own_text = current.own_text.strip()
            _absorb_env_and_choices(current)
            # Register every documented long form so callers can look up an
            # option by any alias it is advertised under.
            for name in (current.long, *current.aliases):
                doc.options.setdefault(name, current)
        current = None
        desc_lines = []

    for index, raw_line in enumerate(lines, start=1):
        sec = _SECTION_RE.match(raw_line.strip())
        if sec:
            flush()
            section = sec.group(1)
            if section and section not in doc.sections:
                doc.sections.append(section)
            continue

        if _is_option_start(raw_line):
            flush()
            usage, tail = split_usage_line(raw_line)
            longs, shorts, remainder = _split_usage(usage)
            if not longs:
                doc.unparsed.append(raw_line)
                continue
            current = OptionDef(
                long=longs[0],
                shorts=tuple(dict.fromkeys(shorts)),
                aliases=tuple(dict.fromkeys(longs[1:])),
                placeholders=_placeholders(remainder),
                choices=_choices_from_usage(remainder),
                section=section,
                line_number=index,
            )
            current.own_text = tail
            if tail:
                desc_lines.append(tail)
            continue

        if current is not None:
            if raw_line.strip():
                desc_lines.append(raw_line.strip())

    flush()
    return doc


def _placeholders(remainder: str) -> Tuple[str, ...]:
    """Extract argument-taking placeholders from a usage remainder."""
    if not remainder:
        return ()
    found: List[str] = []
    for token in remainder.split():
        token = token.strip().rstrip(",")
        if not token:
            continue
        if token.startswith("<") or token.startswith("[") or token.startswith("{"):
            found.append(token)
            continue
        if _PLACEHOLDER_RE.match(token):
            found.append(token)
    return tuple(found)


def _absorb_env_and_choices(opt: OptionDef) -> None:
    """Fill in env var, default text, and allowed values.

    All three are searched across the option's whole help block, not just its
    defining line: upstream prints ``(default: ...)`` and ``allowed values:`` on
    their own indented lines, often several lines below the definition.

    This is safe because :func:`parse_help` terminates an option's block at the
    next definition line (``_is_option_start``), so a following option's text
    can never leak into this one.  That distinction is what keeps ``--no-host``
    from inheriting the ``(default: f16)`` and ``allowed values`` of the
    ``--cache-type-k`` entry that happens to follow it.
    """
    body = opt.help_text
    if not body:
        return

    env = _ENV_RE.search(body)
    if env:
        opt.env = env.group(1)

    allowed = _ALLOWED_RE.search(body)
    if allowed:
        choices = _usable_choices(allowed.group(1))
        if choices:
            opt.choices = choices

    if opt.default_text is None:
        m = _DEFAULT_RE.search(body)
        if m:
            opt.default_text = m.group(1).strip()


def parse_version(text: str) -> str:
    """Return the ``version:`` banner from a ``--version`` dump, if present."""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_commit(build_info: str) -> str:
    """Extract ``b49650adb``-style commit from a version banner."""
    m = re.search(r"commit\s+([0-9a-fA-F]{7,40})", build_info)
    return m.group(1) if m else ""


def build_number(build_info: str) -> str:
    """Extract the numeric build from ``(build 11026, commit ...)``."""
    m = re.search(r"build\s+(\d+)", build_info)
    return m.group(1) if m else ""

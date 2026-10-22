"""Every text-mode file read or write in the package names its encoding.

Without ``encoding=`` Python < 3.15 on Windows uses the locale code page (cp1252),
so designs, predictions or class names with non-ASCII text ("Matérn") fail to save
or come back garbled. CI runs only Linux and macOS, so this walks the source instead.

Checked: ``read_text``/``write_text`` calls, builtin ``open()`` and ``io``/``codecs``/
``gzip``/``bz2``/``lzma`` ``.open()`` whose mode has no "b", and ``path.open("w")``-style
calls with a literal text mode. Python embedded in string constants (generated training
scripts, remote launchers) is parsed and checked the same way, with Jinja tags and
``str.format`` fields stubbed out.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "ai_made_easy"
SKIP = (PACKAGE / "server" / "static",)

# module.open(file, mode, ...) that can open text; binary-only openers are ignored
TEXT_OPENERS = {"io", "codecs", "gzip", "bz2", "lzma"}
BINARY_OPENERS = {"tarfile", "zipfile", "wave", "wavmod", "Image", "webbrowser", "os"}

FILE_IO = re.compile(r"read_text\(|write_text\(|\bopen\(")
JINJA_INLINE_ELSE = re.compile(r"\{%-?\s*(?:else|elif)\b[^%\n]*%\}[^\n]*?\{%-?\s*endif\s*-?%\}")
JINJA_TAG = re.compile(r"\{%.*?%\}")
JINJA_EXPR = re.compile(r"\{\{.*?\}\}")
FORMAT_FIELD = re.compile(r"(?<!\{)\{[A-Za-z_][\w.\[\]]*(?:![rsa])?(?::[^{}]*)?\}(?!\})")


def _jinja(src: str) -> str:
    # tags vanish (keeping line numbers), inline {% else %}..{% endif %} keeps the first branch
    return JINJA_EXPR.sub("_X", JINJA_TAG.sub("", JINJA_INLINE_ELSE.sub("", src)))


def _format(src: str) -> str:
    return FORMAT_FIELD.sub("_X", src).replace("{{", "{").replace("}}", "}")


def _parse_embedded(src: str) -> ast.Module | None:
    for stub in (lambda s: s, _jinja, _format):
        for text in (stub(src), textwrap.dedent(stub(src))):
            try:
                return ast.parse(text)
            except SyntaxError:
                pass
    return None


def _mode(call: ast.Call, index: int) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return call.args[index] if len(call.args) > index else None


def _is_binary(mode: ast.expr | None) -> bool:
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def _needs_encoding(call: ast.Call) -> bool:
    kws = {kw.arg for kw in call.keywords}
    if "encoding" in kws or None in kws:  # None: **kwargs may carry it
        return False
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
        return True
    if isinstance(func, ast.Name) and func.id == "open":
        return not _is_binary(_mode(call, 1))
    if isinstance(func, ast.Attribute) and func.attr == "open":
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        if owner in BINARY_OPENERS:
            return False
        if owner in TEXT_OPENERS:
            return not _is_binary(_mode(call, 1))
        # path.open(mode): only a literal text mode is surely a file (box.open() is a dialog)
        mode = _mode(call, 0)
        return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" not in mode.value
    return False


def _violations(tree: ast.Module, where: str, first_line: int = 1) -> list[str]:
    found = []
    fstring_parts = {id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _needs_encoding(node):
            found.append(f"{where}:{first_line + node.lineno - 1}: {ast.unparse(node)[:90]}")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and "\n" in node.value and FILE_IO.search(node.value) and id(node) not in fstring_parts):
            embedded = _parse_embedded(node.value)
            # count back from the closing quote: an opening '''\ puts line 1 below lineno
            line = first_line + node.end_lineno - 1 - node.value.count("\n")
            if embedded is None:
                found.append(f"{where}:{line}: embedded code does not parse, so it can't be checked")
            else:
                found += _violations(embedded, where, line)
    return found


def _sources() -> list[Path]:
    return [p for p in sorted(PACKAGE.rglob("*.py"))
            if not any(p.is_relative_to(s) for s in SKIP)]


def test_text_io_names_its_encoding():
    found = []
    for path in _sources():
        rel = path.relative_to(PACKAGE.parent).as_posix()
        found += _violations(ast.parse(path.read_text(encoding="utf-8")), rel)
    assert not found, "add encoding=\"utf-8\" to:\n" + "\n".join(found)


def test_checker_catches_each_form():
    src = textwrap.dedent('''
        from pathlib import Path
        Path("a").read_text()
        Path("a").write_text("x")
        open("a")
        open("a", "w")
        open("a", mode="a")
        io.open("a", "r")
        Path("a").open("w")
        open("a", "rb")
        Path("a").open("wb")
        Path("a").read_text(encoding="utf-8")
        open("a", "w", encoding="utf-8")
        tarfile.open(fileobj=buf, mode="r:gz")
        Image.open("a")
        box.open()
        SCRIPT = r"""
        import json
        from pathlib import Path
        x = json.loads(Path({path!r}).read_text())
        """
        TEMPLATE = r"""
        {% if spec.ok %}
        data = Path("{{ spec.path }}").read_text()
        {% endif %}
        """
    ''')
    lines = sorted(int(v.split(":")[1]) for v in _violations(ast.parse(src), "t"))
    assert lines == [3, 4, 5, 6, 7, 8, 9, 20, 24]

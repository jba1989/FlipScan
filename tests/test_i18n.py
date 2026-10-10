"""Display-language switch: the tr() runtime, the /api/language endpoint, and
catalog coverage — every Chinese source string the backend or the GUI
renders must have an English entry, or switching languages silently shows
Chinese again."""
import ast
import json
import re
from pathlib import Path

import pytest

from flipscan import i18n
from flipscan.i18n import tr

PKG = Path(__file__).resolve().parent.parent / "flipscan"
STATIC = PKG / "ui" / "static"
CJK = re.compile(r"[\u4e00-\u9fff]")
PLACEHOLDER = re.compile(r"\{(\d+)[^{}]*\}")


@pytest.fixture
def lang():
    before = i18n.get_language()
    yield i18n.set_language
    i18n.set_language(before)


# ---------------------------------------------------------------- runtime

def test_tr_chinese_returns_source(lang):
    lang("zh-Hant")
    assert tr("工作處理器已停止") == "工作處理器已停止"
    assert tr("找不到影片：{0}", "a.mp4") == "找不到影片：a.mp4"


def test_tr_english_uses_catalog(lang):
    lang("en")
    assert tr("工作處理器已停止") == "worker stopped"
    assert tr("找不到影片：{0}", "a.mp4") == "video not found: a.mp4"


def test_tr_missing_entry_falls_back_to_chinese(lang):
    lang("en")
    assert tr("目錄裡沒有的句子 {0}", 3) == "目錄裡沒有的句子 3"


def test_tr_does_not_format_argument_values(lang):
    lang("en")
    # braces in user data must come through untouched
    assert tr("找不到影片：{0}", "{0}{bad}") == "video not found: {0}{bad}"
    # a template without args is never parsed as a format string
    assert tr("{not a placeholder") == "{not a placeholder"


def test_tr_keeps_conversions_and_specs(lang):
    lang("en")
    assert tr("找不到專案 {0!r}", "x") == "no project 'x'"


@pytest.mark.parametrize("tag, expected", [
    ("zh_TW.UTF-8", "zh-Hant"), ("zh-Hant", "zh-Hant"), ("zh", "zh-Hant"),
    ("en-US", "en"), ("EN", "en"), ("fr_FR", None), ("", None), (None, None),
])
def test_normalize(tag, expected):
    assert i18n.normalize(tag) == expected


def test_set_language_rejects_unknown(lang):
    with pytest.raises(ValueError):
        lang("fr")


def test_env_default(monkeypatch):
    monkeypatch.setattr(i18n, "_lang", None)
    monkeypatch.setenv("FLIPSCAN_LANG", "en_US")
    assert i18n.get_language() == "en"
    monkeypatch.setattr(i18n, "_lang", None)
    monkeypatch.delenv("FLIPSCAN_LANG")
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    assert i18n.get_language() == "zh-Hant"


# ---------------------------------------------------------------- endpoint

def test_language_endpoint(tmp_path, monkeypatch, lang):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from flipscan.ui import create_app
    monkeypatch.setenv("FLIPSCAN_ROOT", str(tmp_path))
    monkeypatch.setenv("FLIPSCAN_EXTERNAL_WORKER", "1")
    client = TestClient(create_app(tmp_path, token="t0k"), headers={"X-FlipScan-Token": "t0k"})
    r = client.put("/api/language", json={"lang": "en"})
    assert r.status_code == 200 and r.json() == {"lang": "en"}
    assert i18n.get_language() == "en"
    r = client.get("/api/projects/nope")
    assert r.status_code == 404 and r.json()["detail"] == "no project 'nope'"
    r = client.put("/api/language", json={"lang": "klingon"})
    assert r.status_code == 400 and "unsupported language" in r.json()["detail"]
    client.put("/api/language", json={"lang": "zh-Hant"})
    assert "找不到專案" in client.get("/api/projects/nope").json()["detail"]


# ---------------------------------------------------------------- backend coverage

def _py_sources():
    return [p for p in PKG.rglob("*.py") if p.name != "i18n.py"]


def _module_constants(tree):
    """CJK string values of module-level assignments (display maps / prefixes,
    translated where they're used)."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for n in ast.walk(node.value):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                        and CJK.search(n.value):
                    out.append(n.value)
    return out


def test_backend_catalog_covers_every_tr_call():
    en = i18n.catalog("en")
    missing = []
    for path in _py_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tr"
                    and n.args and isinstance(n.args[0], ast.Constant)):
                if n.args[0].value not in en:
                    missing.append(f"{path.name}:{n.lineno} {n.args[0].value!r}")
        if path.parent.name != "backends":        # prompts stay as written
            missing += [f"{path.name} constant {s!r}" for s in _module_constants(tree)
                        if s not in en]
    assert not missing, "\n".join(missing)


def test_backend_has_no_bare_chinese_messages():
    """A CJK literal must be tr()'s template, a module-level display constant,
    or model prompt text — anything else is a message that won't switch."""
    bare = []
    for path in _py_sources():
        if path.parent.name == "backends" and path.name == "__init__.py":
            continue                               # the transcription prompt
        tree = ast.parse(path.read_text(encoding="utf-8"))
        ok = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tr":
                if n.args:
                    ok.add(id(n.args[0]))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                ok.update(id(n) for n in ast.walk(node))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and CJK.search(n.value):
                if id(n) not in ok and not _is_docstring(n, tree):
                    bare.append(f"{path.relative_to(PKG)}:{n.lineno} {n.value[:40]!r}")
    assert not bare, "\n".join(bare)


def _is_docstring(node, tree):
    for scope in ast.walk(tree):
        body = getattr(scope, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and body[0].value is node:
            return True
    return False


def test_catalog_placeholders_match():
    """An English entry may drop or reorder placeholders, never invent one."""
    for path in (PKG / "locales" / "en.json",):
        cat = json.loads(path.read_text(encoding="utf-8"))
        for zh, en in cat.items():
            assert set(PLACEHOLDER.findall(en)) <= set(PLACEHOLDER.findall(zh)), zh
            assert not CJK.search(en), zh


# ---------------------------------------------------------------- frontend coverage

def _js_catalog():
    src = (STATIC / "i18n.en.js").read_text(encoding="utf-8")
    return json.loads(src[src.index("{"):src.rindex("}") + 1])


_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "`": "`",
        "$": "$", "0": "\0", "\n": ""}


def _read_escape(s, i):
    """(cooked char, next index) for the escape starting at s[i] == '\\'."""
    c = s[i + 1]
    if c == "u":
        if s[i + 2] == "{":
            end = s.index("}", i)
            return chr(int(s[i + 3:end], 16)), end + 1
        return chr(int(s[i + 2:i + 6], 16)), i + 6
    if c == "x":
        return chr(int(s[i + 2:i + 4], 16)), i + 4
    return _ESC.get(c, c), i + 2


def _skip_string(s, i):
    q, i = s[i], i + 1
    out = []
    while s[i] != q:
        if s[i] == "\\":
            ch, i = _read_escape(s, i)
            out.append(ch)
        else:
            out.append(s[i])
            i += 1
    return "".join(out), i + 1


def _skip_template(s, i):
    """Cooked catalog key ("…{0}…") of the template literal at s[i] == '`'."""
    i += 1
    out, n = [], 0
    while s[i] != "`":
        if s[i] == "\\":
            ch, i = _read_escape(s, i)
            out.append(ch)
        elif s.startswith("${", i):
            i = _skip_code(s, i + 2, "}")
            out.append("{%d}" % n)
            n += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out), i + 1


_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^") | {""}


def _skip_code(s, i, close):
    """Scan JS from i to the matching `close`; return the index after it."""
    depth = 0
    while True:
        c = s[i]
        if c in "\"'":
            i = _skip_string(s, i)[1]
        elif c == "`":
            i = _skip_template(s, i)[1]
        elif s.startswith("//", i):
            i = s.index("\n", i)
        elif s.startswith("/*", i):
            i = s.index("*/", i) + 2
        elif c == "/" and _prev_sig(s, i) in _REGEX_PREV:
            i = _skip_regex(s, i)
        elif c in "({[":
            depth += 1
            i += 1
        elif c in ")}]":
            if depth == 0 and c == close:
                return i + 1
            depth -= 1
            i += 1
        else:
            i += 1


def _prev_sig(s, i):
    j = i - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    if j < 0:
        return ""
    # keywords after which a slash starts a regex
    m = re.search(r"(?<![\w$])(return|typeof|case|in|of)$", s[max(0, j - 7):j + 1])
    return "(" if m else s[j]


def _skip_regex(s, i):
    i += 1
    in_class = False
    while True:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            i += 1
            while s[i].isalpha():
                i += 1
            return i
        i += 1


def _js_keys(script):
    """Every t("…") / T`…` key in a script, plus the script with all literals,
    comments and regexes blanked (to look for bare Chinese)."""
    keys, rest, i = [], [], 0
    while i < len(script):
        c = script[i]
        start = i
        if c in "\"'":
            val, i = _skip_string(script, i)
            if re.search(r"(?<![\w$.])t\(\s*$", script[max(0, start - 8):start]):
                keys.append(val)
            elif CJK.search(val):
                rest.append(val)
            continue
        if c == "`":
            val, i = _skip_template(script, i)
            if re.search(r"(?<![\w$.])T$", script[max(0, start - 2):start]):
                keys.append(val)
            elif CJK.search(val):
                rest.append(val)
            continue
        if script.startswith("//", i):
            i = script.index("\n", i)
            continue
        if script.startswith("/*", i):
            i = script.index("*/", i) + 2
            continue
        if c == "/" and _prev_sig(script, i) in _REGEX_PREV:
            i = _skip_regex(script, i)
            continue
        i += 1
    return keys, rest


def _inline_scripts(html):
    return re.findall(r"<script>([\s\S]*?)</script>", html)


@pytest.mark.parametrize("page", ["index.html", "reader.html"])
def test_frontend_catalog_covers_every_key(page):
    cat = _js_catalog()
    html = (STATIC / page).read_text(encoding="utf-8")
    keys, bare = [], []
    for script in _inline_scripts(html):
        k, b = _js_keys(script)
        keys += k
        bare += b
    assert len(keys) > 10
    missing = [k for k in keys if CJK.search(k) and k not in cat]
    assert not missing, "\n---\n".join(missing)
    assert not bare, "untranslatable Chinese literal(s):\n" + "\n".join(bare)


@pytest.mark.parametrize("page", ["index.html", "reader.html"])
def test_frontend_static_markup_covered(page):
    from html.parser import HTMLParser
    cat = _js_catalog()
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    missing = []

    class P(HTMLParser):
        skip = 0

        def handle_starttag(self, tag, attrs):
            self.skip += tag in ("script", "style")
            for k, v in attrs:
                if k in ("title", "placeholder", "aria-label", "alt") and v \
                        and CJK.search(v) and norm(v) not in cat:
                    missing.append(v)

        def handle_endtag(self, tag):
            self.skip -= tag in ("script", "style")

        def handle_data(self, d):
            if not self.skip and CJK.search(d) and norm(d) not in cat:
                missing.append(d)

    P().feed((STATIC / page).read_text(encoding="utf-8"))
    assert not missing, missing


def test_frontend_catalog_placeholders_match():
    for zh, en in _js_catalog().items():
        assert set(PLACEHOLDER.findall(en)) <= set(PLACEHOLDER.findall(zh)), zh
        assert not CJK.search(en), zh


@pytest.mark.parametrize("page", ["index.html", "reader.html"])
def test_frontend_never_shadows_t(page):
    """A local named t or T would turn t("…") / T`…` into a TypeError in that
    scope — in both languages."""
    binding = re.compile(r"\b(const|let|var|function)\s+[tT]\b|\b[tT]\s*=>"
                         r"|\(\s*[tT]\s*[,)]|,\s*[tT]\s*\)\s*=>")
    for script in _inline_scripts((STATIC / page).read_text(encoding="utf-8")):
        hits = [m.group(0) for m in binding.finditer(script)]
        assert not hits, hits

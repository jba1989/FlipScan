"""CLI backend (codex / claude_cli / agy): subprocess.run is always mocked —
no real CLI is ever called."""
import json
import subprocess
from pathlib import Path

import pytest

from flipscan.backends import PROMPT, get_backend
from flipscan.backends import cli_backend as cb
from flipscan.config import DEFAULTS

GOOD = {"markdown": "hello world", "page_number_printed": 7,
        "confidence": "high", "regions": [], "flags": []}
GOOD_JSON = json.dumps(GOOD)
TRUNCATED = '{"markdown": "partial page text that was cut o'


def make_cfg(name, **over):
    prov = {**DEFAULTS["provider"], "name": name, **over}
    return {**DEFAULTS, "provider": prov}


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "p.jpg"
    p.write_bytes(b"\xff\xd8fakejpeg")
    return p


@pytest.fixture(autouse=True)
def fake_which(monkeypatch):
    monkeypatch.setattr(cb.shutil, "which", lambda n: f"/usr/bin/{Path(n).name}")


@pytest.fixture(autouse=True)
def codex_logged_in(monkeypatch, tmp_path):
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text("{}")
    monkeypatch.setattr(cb, "codex_home", lambda: home)


class Runner:
    """Stands in for subprocess.run; `script` is a list of outcomes per call:
    a str (stdout), an Exception to raise, or a CompletedProcess."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        out = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(out, Exception):
            raise out
        if isinstance(out, subprocess.CompletedProcess):
            return out
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


def claude_out(text, is_error=False):
    return "\n".join([
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "result", "is_error": is_error, "result": text})])


def install(monkeypatch, script):
    r = Runner(script)
    monkeypatch.setattr(cb.subprocess, "run", r)
    return r


# ------------------------------------------------------------ command assembly

def test_codex_command(tmp_path, img):
    (tmp_path / cb.IMAGE_NAME).write_bytes(b"x")
    inv = cb.build_invocation("codex", "/bin/codex", tmp_path, PROMPT, "gpt-x")
    a = inv.argv
    assert a[:2] == ["/bin/codex", "exec"]
    assert a[a.index("-s") + 1] == "read-only"
    assert "--skip-git-repo-check" in a
    assert a[a.index("-i") + 1] == str(tmp_path / cb.IMAGE_NAME)
    assert a[a.index("-o") + 1] == str(tmp_path / cb.OUT_NAME)
    assert a[a.index("-m") + 1] == "gpt-x"
    disabled = {a[i + 1] for i, x in enumerate(a) if x == "--disable"}
    assert {"shell_tool", "unified_exec", "browser_use"} <= disabled
    assert a[-1] == "-" and inv.stdin.startswith(PROMPT)
    assert not any("dangerously" in x for x in a)


def test_claude_command_disables_tools_and_sends_image(tmp_path):
    (tmp_path / cb.IMAGE_NAME).write_bytes(b"jpegbytes")
    inv = cb.build_invocation("claude_cli", "/bin/claude", tmp_path, PROMPT)
    a = inv.argv
    assert a[0] == "/bin/claude" and "-p" in a
    assert a[a.index("--tools") + 1] == ""
    assert a[a.index("--input-format") + 1] == "stream-json"
    assert "--strict-mcp-config" in a
    assert json.loads(a[a.index("--settings") + 1]) == {"disableAllHooks": True}
    assert "--bare" not in a          # would drop the subscription login
    assert "--model" not in a
    assert not any("dangerously" in x for x in a)
    blocks = json.loads(inv.stdin)["message"]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[1]["type"] == "text" and blocks[1]["text"].startswith(PROMPT)


def test_agy_command_is_sandboxed_and_points_at_image(tmp_path):
    (tmp_path / cb.IMAGE_NAME).write_bytes(b"x")
    inv = cb.build_invocation("agy", "/bin/agy", tmp_path, PROMPT, "m1")
    a = inv.argv
    assert "--sandbox" in a and a[a.index("--model") + 1] == "m1"
    assert a[-1].startswith("-p=" + PROMPT) and cb.IMAGE_NAME in a[-1]
    assert inv.stdin is None
    assert not any("dangerously" in x for x in a)


def test_run_uses_argv_list_private_cwd_and_timeout(monkeypatch, img):
    r = install(monkeypatch, [GOOD_JSON])
    b = get_backend(make_cfg("agy", cli_timeout=42))
    assert b.transcribe([("p1", img)], log=lambda m: None)["p1"]["markdown"] == "hello world"
    argv, kw = r.calls[0]
    assert isinstance(argv, list) and not kw.get("shell")
    assert kw["timeout"] == 42
    assert Path(kw["cwd"]) != img.parent          # a throw-away dir
    assert not Path(kw["cwd"]).exists()           # cleaned up afterwards


def test_api_keys_stripped_from_env(monkeypatch, img):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    r = install(monkeypatch, [claude_out(GOOD_JSON)])
    get_backend(make_cfg("claude_cli")).transcribe([("p1", img)], log=lambda m: None)
    assert "ANTHROPIC_API_KEY" not in r.calls[0][1]["env"]


# ------------------------------------------------------------ results

@pytest.mark.parametrize("provider, out", [
    ("claude_cli", claude_out(GOOD_JSON)),
    ("agy", GOOD_JSON),
    ("codex", GOOD_JSON),                      # no -o file written -> stdout
])
def test_good_json_parsed(monkeypatch, img, provider, out):
    install(monkeypatch, [out])
    res = get_backend(make_cfg(provider)).transcribe([("p1", img)], log=lambda m: None)
    assert res["p1"]["markdown"] == "hello world"
    assert res["p1"]["page_number_printed"] == 7


def test_codex_reads_output_file(monkeypatch, img):
    def run(argv, **kw):
        Path(argv[argv.index("-o") + 1]).write_text(GOOD_JSON)
        return subprocess.CompletedProcess(argv, 0, stdout="noise", stderr="")
    monkeypatch.setattr(cb.subprocess, "run", run)
    res = get_backend(make_cfg("codex")).transcribe([("p1", img)], log=lambda m: None)
    assert res["p1"]["markdown"] == "hello world"


def test_truncated_json_is_salvaged(monkeypatch, img):
    install(monkeypatch, [TRUNCATED])
    res = get_backend(make_cfg("agy", cli_retries=0)).transcribe(
        [("p1", img)], log=lambda m: None)
    assert res["p1"]["flags"] == ["truncated"]
    assert res["p1"]["markdown"].startswith("partial page text")


def test_garbage_is_an_error(monkeypatch, img):
    r = install(monkeypatch, ["I cannot help with that."])
    res = get_backend(make_cfg("agy", cli_retries=1)).transcribe(
        [("p1", img)], log=lambda m: None)
    assert "error" in res["p1"]
    assert len(r.calls) == 2                      # retried once


def test_claude_is_error_event(monkeypatch, img):
    install(monkeypatch, [claude_out("Not logged in", is_error=True)])
    res = get_backend(make_cfg("claude_cli", cli_retries=0)).transcribe(
        [("p1", img)], log=lambda m: None)
    assert "Not logged in" in res["p1"]["error"]


# ------------------------------------------------------------ failures / retry

def test_timeout_retries_then_succeeds(monkeypatch, img):
    r = install(monkeypatch, [subprocess.TimeoutExpired("agy", 1),
                              subprocess.TimeoutExpired("agy", 1), GOOD_JSON])
    res = get_backend(make_cfg("agy")).transcribe([("p1", img)], log=lambda m: None)
    assert "error" not in res["p1"] and len(r.calls) == 3


def test_timeout_every_time_records_error(monkeypatch, img):
    r = install(monkeypatch, [subprocess.TimeoutExpired("agy", 1)])
    res = get_backend(make_cfg("agy", cli_retries=2)).transcribe(
        [("p1", img)], log=lambda m: None)
    assert "error" in res["p1"] and len(r.calls) == 3


def test_nonzero_exit_records_stderr(monkeypatch, img):
    bad = subprocess.CompletedProcess([], 3, stdout="", stderr="auth expired")
    r = install(monkeypatch, [bad])
    res = get_backend(make_cfg("codex", cli_retries=1)).transcribe(
        [("p1", img)], log=lambda m: None)
    assert "auth expired" in res["p1"]["error"] and len(r.calls) == 2


def test_failure_of_one_page_does_not_sink_others(monkeypatch, img):
    install(monkeypatch, [GOOD_JSON])
    b = get_backend(make_cfg("agy", cli_concurrency=2))
    res = b.transcribe([("a", img), ("b", img), ("c", img)], log=lambda m: None)
    assert set(res) == {"a", "b", "c"}


def test_missing_executable_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(cb.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="codex"):
        get_backend(make_cfg("codex"))


def test_cli_path_overrides_lookup(monkeypatch):
    seen = []
    monkeypatch.setattr(cb.shutil, "which", lambda n: seen.append(n) or n)
    b = get_backend(make_cfg("agy", cli_path="/opt/agy"))
    assert seen == ["/opt/agy"] and b.exe == "/opt/agy"


# ------------------------------------------------------------ routing / orientation

@pytest.mark.parametrize("name", ["codex", "claude_cli", "agy"])
def test_get_backend_routes(name):
    b = get_backend(make_cfg(name, cli_model="m"))
    assert isinstance(b, cb.CliBackend) and b.provider == name
    assert b.name == f"{name}:m"


def test_defaults_present():
    p = DEFAULTS["provider"]
    assert (p["cli_model"], p["cli_timeout"], p["cli_concurrency"],
            p["cli_retries"], p["cli_path"]) == ("", 300, 1, 2, "")


@pytest.mark.parametrize("out, expected", [
    ('{"upside_down": true}', True), ('{"upside_down": false}', False),
    ("no idea", None)])
def test_check_orientation(monkeypatch, img, out, expected):
    install(monkeypatch, [out])
    assert get_backend(make_cfg("agy")).check_orientation(img) is expected


def test_check_orientation_swallows_failures(monkeypatch, img):
    install(monkeypatch, [subprocess.TimeoutExpired("agy", 1)])
    assert get_backend(make_cfg("agy")).check_orientation(img) is None


# ------------------------------------------------------------ config trust

def test_project_config_cannot_choose_the_executable(tmp_path, monkeypatch):
    """A book folder may come from someone else: its config.toml must not be
    able to point cli_path at an arbitrary program."""
    from flipscan import config
    monkeypatch.setattr(config, "global_config_path",
                        lambda: tmp_path / "global.toml")
    (tmp_path / "global.toml").write_text('[provider]\ncli_path = "/usr/bin/claude"\n')
    book = tmp_path / "book"
    book.mkdir()
    (book / "config.toml").write_text(
        '[provider]\nname = "codex"\ncli_path = "/tmp/evil.sh"\n')
    p = config.load_config(book)["provider"]
    assert p["name"] == "codex"                 # choosing a provider is fine
    assert p["cli_path"] == "/usr/bin/claude"   # the global value survives


def test_project_config_cannot_redirect_endpoints(tmp_path, monkeypatch):
    """Redirecting a URL would ship the global API key (or every page image)
    to whoever wrote the book folder's config.toml."""
    from flipscan import config
    monkeypatch.setattr(config, "global_config_path",
                        lambda: tmp_path / "global.toml")
    book = tmp_path / "book"
    book.mkdir()
    (book / "config.toml").write_text(
        '[provider]\nopenai_base_url = "https://evil.example/v1"\n'
        'ollama_url = "http://evil.example:11434"\n')
    p = config.load_config(book)["provider"]
    assert p["openai_base_url"] == config.DEFAULTS["provider"]["openai_base_url"]
    assert p["ollama_url"] == config.DEFAULTS["provider"]["ollama_url"]


def test_codex_runs_isolated_from_user_config(monkeypatch, tmp_path, img):
    """codex gets its own CODEX_HOME so ~/.codex MCP servers / plugins never load."""
    r = install(monkeypatch, [""])
    get_backend(make_cfg("codex")).transcribe([("p1", img)], log=lambda m: None)
    assert r.calls[0][1]["env"]["CODEX_HOME"] == str(cb.codex_home())


def test_codex_without_its_own_login_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "codex_home", lambda: tmp_path / "empty")
    monkeypatch.setattr(cb.shutil, "which", lambda x: "/bin/codex")
    with pytest.raises(RuntimeError, match="codex login"):
        get_backend(make_cfg("codex"))

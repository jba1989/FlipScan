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
    b = get_backend(make_cfg(name, **{f"{name}_model": "m"}))
    assert isinstance(b, cb.CliBackend) and b.provider == name
    assert b.name == f"{name}:m"


def test_defaults_present():
    p = DEFAULTS["provider"]
    assert (p["cli_timeout"], p["cli_concurrency"],
            p["cli_retries"], p["cli_path"]) == (300, 1, 2, "")
    assert "cli_model" not in p


def test_default_models_are_the_cheapest_that_passed_ocr():
    p = DEFAULTS["provider"]
    assert (p["codex_model"], p["codex_effort"]) == ("gpt-6-luna", "low")
    assert p["claude_cli_model"] == "sonnet"
    assert p["agy_model"] == "gemini-3.8-flash-low"


@pytest.mark.parametrize("name, model", [
    ("codex", "gpt-6-luna"), ("claude_cli", "sonnet"),
    ("agy", "gemini-3.8-flash-low")])
def test_each_provider_uses_its_own_model(monkeypatch, img, name, model):
    r = install(monkeypatch, [claude_out(GOOD_JSON) if name == "claude_cli"
                              else GOOD_JSON])
    b = get_backend(make_cfg(name))
    assert b.model == model
    b.transcribe([("p1", img)], log=lambda m: None)
    a = r.calls[0][0]
    assert a[a.index("-m" if name == "codex" else "--model") + 1] == model


def test_other_providers_model_is_ignored():
    b = get_backend(make_cfg("agy", codex_model="x", claude_cli_model="y"))
    assert b.model == "gemini-3.8-flash-low"


def test_empty_model_means_cli_default(tmp_path):
    (tmp_path / cb.IMAGE_NAME).write_bytes(b"x")
    assert get_backend(make_cfg("agy", agy_model="")).model == ""
    assert "--model" not in cb.build_invocation(
        "agy", "/bin/agy", tmp_path, PROMPT, "").argv


def test_codex_effort_flag(tmp_path):
    (tmp_path / cb.IMAGE_NAME).write_bytes(b"x")
    a = cb.build_invocation("codex", "/bin/codex", tmp_path, PROMPT,
                            "gpt-6-luna", "low").argv
    assert 'model_reasoning_effort="low"' in a
    assert a[a.index('model_reasoning_effort="low"') - 1] == "-c"
    a = cb.build_invocation("codex", "/bin/codex", tmp_path, PROMPT).argv
    assert not any("reasoning_effort" in x for x in a)


def test_effort_only_applies_to_codex(tmp_path):
    (tmp_path / cb.IMAGE_NAME).write_bytes(b"x")
    a = cb.build_invocation("agy", "/bin/agy", tmp_path, PROMPT, "m", "low").argv
    assert not any("reasoning_effort" in x for x in a)


def test_codex_backend_passes_configured_effort(monkeypatch, img):
    r = install(monkeypatch, [GOOD_JSON])
    get_backend(make_cfg("codex", codex_effort="high")).transcribe(
        [("p1", img)], log=lambda m: None)
    assert 'model_reasoning_effort="high"' in r.calls[0][0]


@pytest.mark.parametrize("effort", ["max", 'low"; x', "LOW", " low", "-low"])
def test_rejects_bad_codex_effort(effort):
    with pytest.raises(RuntimeError, match="codex_effort"):
        cb.CliBackend(make_cfg("codex", codex_effort=effort))


def test_empty_codex_effort_is_allowed(tmp_path):
    assert cb.CliBackend(make_cfg("codex", codex_effort="")).effort == ""


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


def test_codex_home_cannot_collide_with_a_project(monkeypatch):
    """codex-home sits next to the book projects. A project of the same name
    would hand codex that folder's config.toml (MCP servers = code execution),
    so the name must be one a project can never have."""
    from flipscan.ui.security import is_plain_name
    monkeypatch.undo()                      # the real codex_home, not the fixture's
    assert not is_plain_name(cb.codex_home().name)


@pytest.mark.parametrize("model", ["--dangerously-skip-permissions", "-m",
                                   "a b", "x;rm -rf", "a" * 81, " gpt-5"])
def test_rejects_unsafe_cli_model(model):
    with pytest.raises(RuntimeError, match="claude_cli_model"):
        cb.CliBackend(make_cfg("claude_cli", claude_cli_model=model))


@pytest.mark.parametrize("model", ["", "gpt-5.1-codex", "claude-sonnet-4-5",
                                   "gemini-3-pro:high", "openai/gpt-4o"])
def test_accepts_plain_cli_model(model):
    assert cb.CliBackend(make_cfg("claude_cli", claude_cli_model=model)).model == model


# ------------------------------------------------------------ model catalog

CODEX_MODELS = [
    {"slug": "gpt-6-luna", "display_name": "GPT-6-Luna", "description": "Small",
     "visibility": "list"},
    {"slug": "hidden-one", "display_name": "Hidden", "description": "",
     "visibility": "hide"},
    {"slug": "gpt-6.1-sol", "display_name": "GPT-6.1-Sol", "description": "Big",
     "visibility": "list"},
]
AGY_OUT = ("Fetching available models...\n"
           "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
           "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n\n")


@pytest.fixture(autouse=True)
def fresh_model_cache():
    cb.MODEL_CACHE.clear()
    yield
    cb.MODEL_CACHE.clear()


def ids(entries):
    return [e["id"] for e in entries]


@pytest.mark.parametrize("shape", ["list", "dict"])
def test_list_models_codex_parses_both_json_shapes(monkeypatch, shape):
    payload = CODEX_MODELS if shape == "list" else {"models": CODEX_MODELS}
    r = install(monkeypatch, [json.dumps(payload)])
    out = cb.list_models("codex")
    assert ids(out) == ["gpt-6-luna", "gpt-6.1-sol"]      # hidden one dropped
    assert out[0]["label"] == "GPT-6-Luna" and out[0]["note"] == "Small"
    assert out[0]["recommended"] is True and not out[1].get("recommended")
    argv, kw = r.calls[0]
    assert argv[1:3] == ["debug", "models"] and isinstance(argv, list)
    assert kw["env"]["CODEX_HOME"] == str(cb.codex_home())
    assert kw["timeout"] <= 30


def test_list_models_codex_retries_without_isolated_home(monkeypatch):
    r = install(monkeypatch, [subprocess.CompletedProcess([], 1, "", "no login"),
                              json.dumps(CODEX_MODELS)])
    assert ids(cb.list_models("codex")) == ["gpt-6-luna", "gpt-6.1-sol"]
    assert len(r.calls) == 2
    assert r.calls[1][1]["env"].get("CODEX_HOME") != str(cb.codex_home())


@pytest.mark.parametrize("script", [
    [subprocess.TimeoutExpired("codex", 1)],
    [subprocess.CompletedProcess([], 1, "", "boom")],
    ["not json"], ["[]"], ['{"models": []}'], ['"x"'], [OSError("gone")]])
def test_list_models_codex_falls_back_to_static(monkeypatch, script):
    install(monkeypatch, script)
    out = cb.list_models("codex")
    assert ids(out) == ["gpt-6-luna", "gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-luna"]
    assert [e["id"] for e in out if e.get("recommended")] == ["gpt-6-luna"]


def test_list_models_agy_parses_tab_lines(monkeypatch):
    r = install(monkeypatch, [AGY_OUT])
    out = cb.list_models("agy")
    assert ids(out) == ["gemini-3.8-flash-high", "gemini-3.8-flash-low"]
    assert out[1]["label"] == "Gemini 3.8 Flash (Low)" and out[1]["recommended"]
    assert r.calls[0][0][1:] == ["models"]


@pytest.mark.parametrize("script", [
    [subprocess.TimeoutExpired("agy", 1)], ["Fetching available models...\n"],
    [subprocess.CompletedProcess([], 2, "", "x")], [OSError("gone")]])
def test_list_models_agy_falls_back_to_static(monkeypatch, script):
    install(monkeypatch, script)
    assert ids(cb.list_models("agy")) == ["gemini-3.8-flash-low",
                                          "gemini-3.8-flash-medium"]


def test_list_models_claude_is_static_and_never_spawns(monkeypatch):
    r = install(monkeypatch, [""])
    out = cb.list_models("claude_cli")
    assert ids(out) == ["haiku", "sonnet", "opus"] and not r.calls
    assert [e["id"] for e in out if e.get("recommended")] == ["sonnet"]


def test_list_models_missing_cli_falls_back(monkeypatch):
    monkeypatch.setattr(cb.shutil, "which", lambda n: None)
    assert ids(cb.list_models("agy")) == ["gemini-3.8-flash-low",
                                          "gemini-3.8-flash-medium"]


def test_list_models_is_cached(monkeypatch):
    r = install(monkeypatch, [AGY_OUT])
    cb.list_models("agy")
    cb.list_models("agy")
    assert len(r.calls) == 1


def test_list_models_drops_unsafe_ids(monkeypatch):
    bad = [{"slug": "--evil", "display_name": "x", "visibility": "list"},
           {"slug": "ok-model", "display_name": "ok", "visibility": "list"}]
    install(monkeypatch, [json.dumps(bad)])
    assert ids(cb.list_models("codex")) == ["ok-model"]

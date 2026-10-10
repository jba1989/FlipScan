"""Security tests for the web GUI: access token, path containment, settings."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from flipscan.ui import create_app  # noqa: E402
from flipscan.ui.security import within  # noqa: E402

TOKEN = "test-token-123"
AUTH = {"X-FlipScan-Token": TOKEN}


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "books"
    r.mkdir()
    # keep config.toml inside the temp root, and no background worker thread
    monkeypatch.setenv("FLIPSCAN_ROOT", str(r))
    monkeypatch.setenv("FLIPSCAN_EXTERNAL_WORKER", "1")
    for k in ("FLIPSCAN_TOKEN", "OPENAI_API_KEY", "FLIPSCAN_OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return r


@pytest.fixture
def client(root):
    return TestClient(create_app(root, token=TOKEN))


# ---------------- path containment

def test_within_rejects_sibling_prefix(tmp_path):
    base = tmp_path / "books"
    assert within(base, base / "a" / "b")
    assert not within(base, tmp_path / "books2" / "x")
    assert not within(base, base / ".." / "etc")


# ---------------- access token

def test_remote_request_without_token_is_rejected(client):
    assert client.get("/api/projects").status_code == 401


def test_wrong_token_is_rejected(client):
    r = client.get("/api/projects", headers={"X-FlipScan-Token": "nope"})
    assert r.status_code == 401


def test_header_token_is_accepted(client):
    assert client.get("/api/projects", headers=AUTH).status_code == 200


def test_query_token_sets_cookie_and_strips_url(client):
    r = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303
    assert "token=" not in r.headers["location"]
    assert "httponly" in r.headers["set-cookie"].lower()
    # the cookie alone now authenticates
    assert client.get("/api/projects").status_code == 200


def test_loopback_browser_needs_no_token(root):
    c = TestClient(create_app(root, token=TOKEN),
                   base_url="http://localhost:8321", client=("127.0.0.1", 5000))
    assert c.get("/api/projects").status_code == 200


def test_loopback_with_foreign_host_header_is_rejected(root):
    # DNS rebinding: the socket is loopback but the browser thinks it's evil.com
    c = TestClient(create_app(root, token=TOKEN),
                   base_url="http://evil.example:8321", client=("127.0.0.1", 5000))
    assert c.get("/api/projects").status_code == 401


def test_token_is_persisted_across_restarts(root):
    from flipscan.ui.security import load_or_create_token
    t1 = load_or_create_token(root)
    assert t1 == load_or_create_token(root)
    assert (root / ".ui_token").stat().st_mode & 0o077 == 0


# ---------------- project creation

@pytest.mark.parametrize("bad", ["../evil", "/tmp/evil", "a/b", ".hidden", ".."])
def test_project_name_must_be_a_plain_slug(client, root, bad):
    r = client.post("/api/projects", headers=AUTH,
                    json={"name": bad, "title": "t", "videos": []})
    assert r.status_code == 400
    assert not (root.parent / "evil").exists()


def test_project_video_must_come_from_uploads(client, root, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    r = client.post("/api/projects", headers=AUTH,
                    json={"title": "t", "videos": [{"path": str(secret)}]})
    assert r.status_code == 400
    assert not any(p.name == "t" for p in root.iterdir())


def test_project_creation_still_works(client):
    r = client.post("/api/projects", headers=AUTH, json={"title": "My Book"})
    assert r.status_code == 200 and r.json()["name"] == "my-book"


def test_cjk_title_creates_project(client):
    r = client.post("/api/projects", headers=AUTH, json={"title": "我的書"})
    assert r.status_code == 200 and r.json()["name"] == "我的書"


def test_existing_non_slug_folder_stays_reachable(client, root):
    # folders made by the CLI (`flipscan init MyBook`) predate the GUI's slugs
    (root / "MyBook").mkdir()
    (root / "MyBook" / "manifest.json").write_text(
        '{"version": 1, "book": {}, "videos": [], "stages": {}, "pages": []}')
    assert client.get("/api/projects/MyBook/jobs", headers=AUTH).status_code == 200


def test_dot_segments_cannot_escape_servable_dirs(client, root):
    client.post("/api/projects", headers=AUTH, json={"title": "b"})
    (root / "b" / "config.toml").write_text('openai_api_key = "sk-secret"')
    r = client.get("/api/projects/b/file/videos/%2E%2E/config.toml", headers=AUTH)
    assert r.status_code in (403, 404)
    assert "sk-secret" not in r.text


# ---------------- settings / outbound requests

def _put_settings(client, **kw):
    body = {"provider": "openai", "openai_model": "gpt-4o", **kw}
    return client.put("/api/settings", headers=AUTH, json=body)


def test_changing_base_url_requires_reentering_key(client, root):
    assert _put_settings(client, openai_base_url="https://api.openai.com/v1",
                         openai_api_key="sk-original").status_code == 200
    r = _put_settings(client, openai_base_url="https://attacker.example/v1")
    assert r.status_code == 400
    assert "sk-original" in (root / "config.toml").read_text()
    assert "attacker" not in (root / "config.toml").read_text()


def test_unchanged_base_url_keeps_stored_key(client, root):
    _put_settings(client, openai_base_url="https://api.openai.com/v1",
                  openai_api_key="sk-original")
    assert _put_settings(client, openai_base_url="https://api.openai.com/v1"
                         ).status_code == 200
    assert "sk-original" in (root / "config.toml").read_text()


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x", "localhost:11434"])
def test_ollama_probe_only_allows_http(client, url):
    r = client.get("/api/settings/ollama-models", params={"url": url}, headers=AUTH)
    assert r.status_code == 400


# ---------------- CSRF against the localhost exemption

@pytest.fixture
def local(root):
    return TestClient(create_app(root, token=TOKEN),
                      base_url="http://localhost:8321", client=("127.0.0.1", 5000))


def test_local_same_origin_request_is_allowed(local):
    r = local.get("/api/projects", headers={"Sec-Fetch-Site": "same-origin",
                                            "Origin": "http://localhost:8321"})
    assert r.status_code == 200


def test_local_cross_site_fetch_is_rejected(local):
    # a page on evil.example making the user's browser call localhost
    r = local.post("/api/jobs/1/cancel", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 401


def test_local_foreign_origin_is_rejected(local):
    # older browsers without Sec-Fetch-* still send Origin on cross-site POSTs
    r = local.post("/api/jobs/1/cancel", headers={"Origin": "https://evil.example"})
    assert r.status_code == 401


def test_local_other_port_origin_is_rejected(local):
    # another dev server on localhost is "same-site" but not this app
    r = local.post("/api/jobs/1/cancel", headers={"Sec-Fetch-Site": "same-site",
                                                  "Origin": "http://localhost:3000"})
    assert r.status_code == 401


# ---------------- env-sourced keys, proxies

def test_base_url_guard_covers_env_key(client, monkeypatch):
    # a key that only lives in the environment must not follow a new URL either
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    r = _put_settings(client, openai_base_url="https://attacker.example/v1")
    assert r.status_code == 400


@pytest.mark.parametrize("hdr", ["X-Forwarded-For", "Forwarded", "X-Real-IP",
                                 "CF-Connecting-IP"])
def test_proxied_request_gets_no_localhost_exemption(local, hdr):
    # a tunnel/reverse proxy on this machine makes remote users look loopback
    assert local.get("/api/projects", headers={hdr: "1.2.3.4"}).status_code == 401


def test_localhost_exemption_can_be_disabled(root, monkeypatch):
    monkeypatch.setenv("FLIPSCAN_REQUIRE_TOKEN", "1")
    c = TestClient(create_app(root, token=TOKEN),
                   base_url="http://localhost:8321", client=("127.0.0.1", 5000))
    assert c.get("/api/projects").status_code == 401
    assert c.get("/api/projects", headers=AUTH).status_code == 200


# ---------------- subscription CLI settings

def _put(client, **over):
    body = {"provider": "mock", **over}
    return client.put("/api/settings", json=body, headers=AUTH)


def test_get_settings_reports_cli_fields(client, root):
    s = client.get("/api/settings", headers=AUTH).json()
    assert "cli_model" not in s
    assert (s["codex_model"], s["codex_effort"]) == ("gpt-6-luna", "low")
    assert s["claude_cli_model"] == "sonnet"
    assert s["agy_model"] == "gemini-3.8-flash-low"
    assert s["cli_concurrency"] == 1
    assert s["cli_timeout"] == 300
    assert s["codex_logged_in"] is False
    assert s["codex_login_cmd"].startswith("CODEX_HOME=")
    assert s["codex_login_cmd"].endswith(" codex login")
    assert str(root) in s["codex_login_cmd"]


def test_codex_logged_in_follows_auth_json(client, root):
    home = root / ".codex-home"
    home.mkdir()
    (home / "auth.json").write_text("{}")
    assert client.get("/api/settings", headers=AUTH).json()["codex_logged_in"] is True


@pytest.mark.parametrize("over", [
    {"codex_model": "--dangerously-skip-permissions"},
    {"claude_cli_model": "a b"}, {"agy_model": "-x"},
    {"codex_effort": "max"}, {"codex_effort": "low; x"},
    {"cli_concurrency": 0}, {"cli_concurrency": 5},
    {"cli_timeout": 59}, {"cli_timeout": 1801},
])
def test_put_rejects_bad_cli_values(client, over):
    r = _put(client, **over)
    assert r.status_code in (400, 422)
    # nothing was written
    assert client.get("/api/settings", headers=AUTH).json()["provider"] != "mock"


@pytest.mark.parametrize("field", ["codex_model", "claude_cli_model",
                                   "agy_model", "codex_effort"])
def test_put_bad_cli_value_names_the_field(client, field):
    assert field in _put(client, **{field: "--x"}).json()["detail"]


def test_put_persists_cli_values_and_keeps_the_rest(client, root):
    (root / "config.toml").write_text(
        '[provider]\ncli_path = "/opt/bin/claude"\ncli_retries = 4\n')
    assert _put(client, codex_model="gpt-6.1-sol", codex_effort="high",
                claude_cli_model="opus", agy_model="gemini-3.8-flash-medium",
                cli_concurrency=2, cli_timeout=600).status_code == 200
    s = client.get("/api/settings", headers=AUTH).json()
    assert (s["codex_model"], s["codex_effort"], s["claude_cli_model"],
            s["agy_model"]) == ("gpt-6.1-sol", "high", "opus",
                                "gemini-3.8-flash-medium")
    assert (s["cli_concurrency"], s["cli_timeout"]) == (2, 600)
    text = (root / "config.toml").read_text()
    assert 'cli_path = "/opt/bin/claude"' in text and "cli_retries = 4" in text
    assert 'codex_model = "gpt-6.1-sol"' in text and "\ncli_model" not in text


def test_put_without_cli_fields_keeps_current(client):
    _put(client, codex_model="gpt-5", agy_model="g", cli_concurrency=3,
         cli_timeout=900)
    _put(client)
    s = client.get("/api/settings", headers=AUTH).json()
    assert (s["codex_model"], s["agy_model"], s["cli_concurrency"],
            s["cli_timeout"]) == ("gpt-5", "g", 3, 900)
    assert s["claude_cli_model"] == "sonnet"      # untouched default


def test_put_empty_cli_model_reverts_to_default(client):
    """save_global_config drops empty values, so "" falls back to the default."""
    _put(client, claude_cli_model="opus")
    _put(client, claude_cli_model="")
    assert client.get("/api/settings", headers=AUTH).json()["claude_cli_model"] == "sonnet"


def test_cli_models_endpoint_shape(client, monkeypatch):
    import flipscan.ui as ui
    monkeypatch.setattr(ui, "list_models", lambda p: [
        {"id": f"{p}-m", "label": p.upper(), "note": "n", "recommended": True}])
    r = client.get("/api/settings/cli-models", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {p: [{"id": f"{p}-m", "label": p.upper(), "note": "n",
                             "recommended": True}]
                        for p in ("codex", "claude_cli", "agy")}
    assert client.get("/api/settings/cli-models").status_code == 401

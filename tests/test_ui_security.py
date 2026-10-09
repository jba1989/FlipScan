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

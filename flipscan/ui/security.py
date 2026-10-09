"""Access control for the web GUI.

The GUI listens on the LAN so phones can reach it, which means anyone on the
same network can too. Every request must therefore carry the per-install
access token (cookie, header, or a one-time ?token= in the URL) — except a
browser on this very machine talking to it as localhost.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

TOKEN_FILE = ".ui_token"
COOKIE = "flipscan_token"
HEADER = "x-flipscan-token"
_COOKIE_MAX_AGE = 400 * 24 * 3600          # browsers cap cookies at ~400 days
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# set by reverse proxies / tunnels (nginx, ngrok, cloudflared, Tailscale serve):
# their traffic reaches us from loopback but comes from someone else
_PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "cf-connecting-ip")


def is_plain_name(name: str) -> bool:
    """One ordinary directory name: no separators, not hidden, not . or ..
    Any charset is fine — _slugify keeps CJK/accented titles as-is."""
    return (bool(name) and "/" not in name and "\\" not in name
            and not name.startswith(".") and Path(name).name == name)


def within(base: Path, target: Path) -> bool:
    """True when `target` resolves inside `base` — a real path check, unlike a
    string prefix test that lets /data/books2 pass for /data/books."""
    return Path(target).resolve().is_relative_to(Path(base).resolve())


def load_or_create_token(root: Path) -> str:
    """The GUI access token: $FLIPSCAN_TOKEN, else one generated once and kept
    in <root>/.ui_token (owner-only) so bookmarked phone URLs survive restarts."""
    env = os.environ.get("FLIPSCAN_TOKEN", "").strip()
    if env:
        return env
    path = Path(root) / TOKEN_FILE
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return token


def _is_local_browser(request: Request) -> bool:
    """Loopback socket AND a localhost Host header. Checking Host as well stops
    DNS rebinding, where a web page re-points its own hostname at 127.0.0.1.
    Off entirely with FLIPSCAN_REQUIRE_TOKEN=1 (e.g. behind a tunnel that adds
    no forwarding headers)."""
    if os.environ.get("FLIPSCAN_REQUIRE_TOKEN", "").strip() not in ("", "0"):
        return False
    if any(h in request.headers for h in _PROXY_HEADERS):
        return False
    client = request.client.host if request.client else ""
    try:
        loopback = ipaddress.ip_address(client).is_loopback
    except ValueError:
        return False
    host = request.headers.get("host", "")
    hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host[1:host.find("]")]
    return loopback and hostname.lower() in _LOCAL_HOSTS and _is_same_origin(request, host)


def _is_same_origin(request: Request, host: str) -> bool:
    """Reject CSRF: a page on another site (or another localhost port) making
    this user's browser call us. Browsers mark such calls via Sec-Fetch-Site,
    and older ones still send a foreign Origin on cross-site POSTs. Non-browser
    clients (curl, scripts) send neither and stay allowed."""
    site = request.headers.get("sec-fetch-site")
    if site and site not in ("same-origin", "none"):
        return False
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc.lower() != host.lower():
        return False
    return True


def _matches(candidate: str | None, token: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), token.encode())


def _denied(request: Request):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "FlipScan access token required"}, 401)
    return PlainTextResponse(
        "FlipScan: access token required.\n\nOpen the link with ?token=... that "
        "`flipscan ui` printed in its terminal (or read <projects root>/.ui_token).",
        401)


def install_token_auth(app: FastAPI, token: str) -> None:
    """Gate every route behind `token` (see module docstring)."""

    @app.middleware("http")
    async def _require_token(request: Request, call_next):
        if _is_local_browser(request):
            return await call_next(request)
        if (_matches(request.cookies.get(COOKIE), token)
                or _matches(request.headers.get(HEADER), token)):
            return await call_next(request)
        if _matches(request.query_params.get("token"), token):
            # first visit from a shared link: swap the URL token for a cookie
            # so it doesn't linger in history, then strip it from the address
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
            url = request.url.path + (f"?{urlencode(rest)}" if rest else "")
            resp = RedirectResponse(url, status_code=303)
            resp.set_cookie(COOKIE, token, max_age=_COOKIE_MAX_AGE, httponly=True,
                            samesite="lax", secure=request.url.scheme == "https")
            return resp
        return _denied(request)

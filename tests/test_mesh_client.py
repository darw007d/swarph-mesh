"""Tests for MeshClient — offline only (mocked HTTP).

Live falsifiability gate against the real lab-OVH gateway lives in
``test_smoke_mesh.py``, gated on ``MESH_GATEWAY_TOKEN`` env.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from swarph_mesh import (
    MeshAuthError,
    MeshClient,
    MeshGatewayError,
    MeshMessage,
    MeshPeer,
    MeshSecretLeakError,
)
from swarph_mesh.mesh_client import _check_no_secret


@pytest.fixture(autouse=True)
def _seed_registry_cache():
    """Offline tests have no real ``/peers`` endpoint. send() now validates the
    recipient FAIL-CLOSED (strict=True), so seed the registry TTL cache with the
    canonical peers these tests use — the check then resolves without a network
    hit. Tests exercising the cold-cache / unreachable path clear it explicitly.
    """
    import time
    from swarph_shared import peer_registry
    peer_registry._cache["names"] = frozenset(
        {"droplet", "lab-ovh", "gridiron", "science-claude",
         "workstation-lc", "gemini-researcher", "gpt-ovh"}
    )
    peer_registry._cache["fetched_at"] = time.time()
    yield
    peer_registry._clear_cache()


# ---------------------------------------------------------------------------
# Helpers — httpx MockTransport for offline tests
# ---------------------------------------------------------------------------


def _client_with_handler(handler) -> MeshClient:
    """Build a MeshClient whose underlying httpx.AsyncClient uses a
    MockTransport handler. Bypasses the lazy _ensure_client by
    constructing the AsyncClient directly with the same shape.
    """
    transport = httpx.MockTransport(handler)
    c = MeshClient(node="lab-ovh", token="test-token", validate_self_name=False)
    c._client = httpx.AsyncClient(
        base_url=c._gateway_url,
        headers={"Authorization": f"Bearer {c._token}"},
        timeout=c._timeout,
        transport=transport,
    )
    return c


# ---------------------------------------------------------------------------
# Credential leak detector
# ---------------------------------------------------------------------------


def test_check_no_secret_passes_clean_text():
    _check_no_secret("just regular content here")  # no raise


def test_check_no_secret_catches_pypi_token():
    with pytest.raises(MeshSecretLeakError, match="pypi token"):
        _check_no_secret("upload with pypi-AgEIcHlwaS5vcmcCJDlhMTJjNTc2X" + "Y" * 30)


def test_check_no_secret_catches_anthropic_key():
    with pytest.raises(MeshSecretLeakError, match="anthropic"):
        _check_no_secret("api key sk-ant-XXXXXXXXXXXXXXXXXXXXX1234567890")


def test_check_no_secret_catches_github_token():
    with pytest.raises(MeshSecretLeakError, match="github"):
        _check_no_secret("ghp_" + "X" * 40)


def test_check_no_secret_catches_jwt():
    jwt = "eyJabcdefghijklmno0.eyJabcdefghijklmno0.signature_part_xxxxxxxx"
    with pytest.raises(MeshSecretLeakError, match="jwt"):
        _check_no_secret(jwt)


def test_check_no_secret_skips_short_jwt_lookalikes():
    """`eyJ.eyJ.x` shouldn't fire — too short to be a real JWT."""
    _check_no_secret("eyJ.eyJ.x")  # no raise


def test_check_no_secret_catches_openai_admin_key_with_admin_label(monkeypatch):
    """v0.6.0 — OpenAI admin keys (``sk-admin-...``) match a NAMED pattern
    that flags the highest-privilege class with explicit error text.
    Pre-fix the catch-all openai-key pattern matched but understated the
    privilege boundary.

    Endpoint that mints these: POST /v1/organization/admin_api_keys.
    Admin keys can create more admin keys, delete keys, and manage
    organization settings — substantially larger blast radius than
    regular sk- or sk-proj- keys.
    """
    admin_key = "sk-admin-" + "X" * 30
    with pytest.raises(MeshSecretLeakError, match="ADMIN"):
        _check_no_secret(admin_key)


def test_check_no_secret_distinguishes_admin_from_regular_openai_key():
    """Admin and regular keys both raise, but the error text differs so
    operators see the privilege class clearly."""
    # Regular project key fires "openai api key" (no ADMIN match)
    proj_key = "sk-proj-" + "X" * 30
    with pytest.raises(MeshSecretLeakError, match="openai api key"):
        _check_no_secret(proj_key)
    # Admin key fires the named ADMIN pattern instead
    admin_key = "sk-admin-" + "Y" * 30
    with pytest.raises(MeshSecretLeakError, match="ADMIN"):
        _check_no_secret(admin_key)


def test_check_no_secret_catches_xai_key():
    """v0.5.1 added xai- pattern; regression test that it still fires."""
    with pytest.raises(MeshSecretLeakError, match="xai"):
        _check_no_secret("xai-" + "Z" * 50)


# ---------------------------------------------------------------------------
# list_peers
# ---------------------------------------------------------------------------


def test_list_peers_unwraps_envelope():
    def handler(request):
        assert request.url.path == "/peers"
        return httpx.Response(200, json={"peers": [
            {"name": "lab-ovh", "url": "http://lab-ovh:8787",
             "capabilities": {"runtime": "claude"},
             "enabled": True},
            {"name": "droplet", "url": "http://droplet:8787",
             "capabilities": {}, "enabled": True},
        ]})

    client = _client_with_handler(handler)
    peers = asyncio.run(client.list_peers())
    asyncio.run(client.aclose())
    assert len(peers) == 2
    assert peers[0].name == "lab-ovh"
    assert isinstance(peers[0], MeshPeer)
    assert peers[1].name == "droplet"


def test_list_peers_handles_bare_list_payload():
    """Some gateway versions return bare list — same as swarph_shared.peer_registry."""
    def handler(request):
        return httpx.Response(200, json=[{"name": "x", "enabled": True}])

    client = _client_with_handler(handler)
    peers = asyncio.run(client.list_peers())
    asyncio.run(client.aclose())
    assert len(peers) == 1
    assert peers[0].name == "x"


def test_list_peers_accepts_unknown_extra_fields():
    """Forward-compat: gateway adds fields, MeshPeer doesn't reject."""
    def handler(request):
        return httpx.Response(200, json={"peers": [{
            "name": "future-peer",
            "enabled": True,
            "future_field_2027": "some-value",
            "another_extra": {"nested": True},
        }]})

    client = _client_with_handler(handler)
    peers = asyncio.run(client.list_peers())
    asyncio.run(client.aclose())
    assert peers[0].name == "future-peer"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def test_fetch_uses_self_node_as_to_node_default():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"messages": [], "n": 0})

    client = _client_with_handler(handler)
    asyncio.run(client.fetch())
    asyncio.run(client.aclose())
    # v0.5.1 wire-shape fix: gateway accepts ?to= NOT ?to_node=.
    # The latter is silently ignored, returning ALL recent messages.
    assert captured["params"]["to"] == "lab-ovh"
    assert "to_node" not in captured["params"]
    assert "unread_only" not in captured["params"]


def test_fetch_passes_unread_only_and_limit():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"messages": [], "n": 0})

    client = _client_with_handler(handler)
    asyncio.run(client.fetch(unread_only=True, limit=5))
    asyncio.run(client.aclose())
    assert captured["params"]["unread_only"] == "true"
    assert captured["params"]["limit"] == "5"


def test_fetch_returns_typed_messages():
    def handler(request):
        return httpx.Response(200, json={"messages": [
            {"id": 1, "from_node": "droplet", "to_node": "lab-ovh",
             "kind": "fyi", "content": "hi", "created_at": "2026-05-08T12:00:00Z",
             "read_at": None},
        ], "n": 1})

    client = _client_with_handler(handler)
    msgs = asyncio.run(client.fetch())
    asyncio.run(client.aclose())
    assert len(msgs) == 1
    assert isinstance(msgs[0], MeshMessage)
    assert msgs[0].id == 1
    assert msgs[0].from_node == "droplet"
    assert msgs[0].kind == "fyi"


def test_fetch_explicit_to_node_override():
    """Caller can peek at a different inbox if they have legitimate reason."""
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"messages": [], "n": 0})

    client = _client_with_handler(handler)
    asyncio.run(client.fetch(to_node="droplet"))
    asyncio.run(client.aclose())
    # Wire shape uses ?to=; kwarg name `to_node` preserved for caller API.
    assert captured["params"]["to"] == "droplet"


def test_fetch_filters_outbound_self_messages_when_querying_own_inbox():
    """v0.5.1 defense-in-depth: even if gateway returns from_node==self
    rows (latent ?to_node= bug pre-fix, or any future quirk), client
    filters them out when target == self.node. Regression-tested."""
    def handler(request):
        return httpx.Response(200, json={"messages": [
            {"id": 1, "from_node": "droplet", "to_node": "lab-ovh",
             "kind": "fyi", "content": "real inbound",
             "created_at": "2026-05-08T12:00:00Z", "read_at": None},
            {"id": 2, "from_node": "lab-ovh", "to_node": "droplet",
             "kind": "fyi", "content": "MY outbound — should be filtered",
             "created_at": "2026-05-08T12:01:00Z", "read_at": None},
        ], "n": 2})

    client = _client_with_handler(handler)
    msgs = asyncio.run(client.fetch())
    asyncio.run(client.aclose())
    assert len(msgs) == 1
    assert msgs[0].from_node == "droplet"


def test_fetch_does_NOT_filter_when_peeking_at_other_inbox():
    """Explicit to_node= override → caller asked for raw view; no client-side
    self-filter applies."""
    def handler(request):
        return httpx.Response(200, json={"messages": [
            {"id": 1, "from_node": "lab-ovh", "to_node": "droplet",
             "kind": "fyi", "content": "lab-to-droplet message",
             "created_at": "2026-05-08T12:00:00Z", "read_at": None},
        ], "n": 1})

    client = _client_with_handler(handler)
    msgs = asyncio.run(client.fetch(to_node="droplet"))
    asyncio.run(client.aclose())
    assert len(msgs) == 1  # not filtered out


# ---------------------------------------------------------------------------
# send — recipient validation + secret guard + payload shape
# ---------------------------------------------------------------------------


def test_send_kind_literal_runtime_guard_rejects_unknown():
    """v0.5.1 (drop DM #722): kind=Literal enum + runtime guard. Client-side
    fail-fast on bad kind so callers don't round-trip a 400 from the gateway.
    """
    from swarph_mesh import MeshClient

    client = _client_with_handler(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="not a valid mesh-gateway"):
        asyncio.run(
            client.send(to="droplet", kind="review_request", content="x")
        )
    asyncio.run(client.aclose())


def test_send_response_with_no_content_field_is_accepted():
    """v0.5.1 (drop DM #722): MeshMessage.content is now Optional[str]=None
    so success POST responses (which omit content) parse cleanly. Pre-fix
    raised pydantic ValidationError on every successful send."""
    def handler(request):
        # Mirror gateway's actual POST /messages success shape — no
        # content field returned.
        return httpx.Response(200, json={
            "id": 99,
            "from_node": "lab-ovh",
            "to_node": "droplet",
            "kind": "fyi",
            "thread_id": None,
            "created_at": "2026-05-08T20:00:00Z",
        })

    client = _client_with_handler(handler)
    msg = asyncio.run(client.send(to="droplet", kind="fyi", content="hello"))
    asyncio.run(client.aclose())
    assert msg.id == 99
    assert msg.from_node == "lab-ovh"
    assert msg.to_node == "droplet"
    assert msg.content is None  # gateway didn't echo content; that's OK


# ---------------------------------------------------------------------------
# send — original tests below
# ---------------------------------------------------------------------------


def test_send_validates_recipient_name():
    """Invalid (non-canonical) recipient → ValueError BEFORE the POST.

    The regex check is first in validate_node_name (strict-independent), so a
    malformed name is rejected regardless of registry reachability.
    """
    client = MeshClient(node="lab-ovh", token="t", validate_self_name=False)
    with pytest.raises(ValueError, match="naming convention"):
        asyncio.run(client.send(to="Bad-Name", kind="fyi", content="x"))


def test_send_fails_closed_when_recipient_unverifiable():
    """HIGH regression (auth-bypass, mesh_client.py:392): when the recipient
    cannot be verified against the registry (cold cache + gateway unreachable),
    send must RAISE — not fail open and POST the DM to a possibly-void name."""
    import time
    from swarph_shared import peer_registry, GatewayUnreachableError
    peer_registry._clear_cache()  # cold: force a real /peers attempt
    client = MeshClient(
        node="lab-ovh", token="t", validate_self_name=False,
        gateway_url="http://127.0.0.1:9",  # refuses fast; unreachable /peers
    )
    with pytest.raises(GatewayUnreachableError):
        asyncio.run(client.send(
            to="ghost-but-regex-valid", kind="fyi", content="x"))


def test_send_resolves_known_alias():
    """`drop` should resolve to canonical `droplet` via swarph_shared.KNOWN_ALIASES."""
    captured = {}

    def handler(request):
        captured["body"] = request.read()
        import json
        captured["json"] = json.loads(captured["body"])
        return httpx.Response(200, json={
            "id": 1, "from_node": "lab-ovh", "to_node": "droplet",
            "kind": "fyi", "content": "x", "created_at": "2026-05-08T12:00:00Z",
            "read_at": None,
        })

    client = _client_with_handler(handler)
    asyncio.run(client.send(to="drop", kind="fyi", content="x"))
    asyncio.run(client.aclose())
    # drop → droplet via KNOWN_ALIASES
    assert captured["json"]["to_node"] == "droplet"


def test_send_refuses_credential_in_content():
    client = MeshClient(node="lab-ovh", token="t", validate_self_name=False)
    leaky = "auth header: Bearer ghp_" + "X" * 40
    with pytest.raises(MeshSecretLeakError):
        asyncio.run(client.send(to="droplet", kind="fyi", content=leaky))


def test_send_skip_secret_check_bypasses_guard():
    """Operator escape hatch when content legitimately contains
    credential-shape prose."""
    captured = {}

    def handler(request):
        import json
        captured["json"] = json.loads(request.read())
        return httpx.Response(200, json={
            "id": 1, "from_node": "lab-ovh", "to_node": "droplet",
            "kind": "fyi", "content": "...", "created_at": "2026-05-08T12:00:00Z",
            "read_at": None,
        })

    client = _client_with_handler(handler)
    leaky = "discussing pypi-AgEIcHlwaS5vcmcCJDlhMTJjNTc2X" + "Y" * 30
    asyncio.run(client.send(to="droplet", kind="fyi", content=leaky, skip_secret_check=True))
    asyncio.run(client.aclose())
    assert captured["json"]["content"] == leaky


def test_send_returns_typed_message():
    def handler(request):
        return httpx.Response(200, json={
            "id": 42, "from_node": "lab-ovh", "to_node": "droplet",
            "kind": "fyi", "content": "x", "created_at": "2026-05-08T12:00:00Z",
            "read_at": None,
        })

    client = _client_with_handler(handler)
    sent = asyncio.run(client.send(to="droplet", kind="fyi", content="x"))
    asyncio.run(client.aclose())
    assert isinstance(sent, MeshMessage)
    assert sent.id == 42
    assert sent.to_node == "droplet"


def test_send_passes_optional_fields():
    captured = {}

    def handler(request):
        import json
        captured["json"] = json.loads(request.read())
        return httpx.Response(200, json={
            "id": 1, "from_node": "lab-ovh", "to_node": "droplet",
            "kind": "answer", "content": "x", "created_at": "2026-05-08T12:00:00Z",
            "read_at": None,
        })

    client = _client_with_handler(handler)
    asyncio.run(client.send(
        to="droplet", kind="answer", content="x",
        related_task_id="T-1", thread_id="thr-abc",
    ))
    asyncio.run(client.aclose())
    assert captured["json"]["related_task_id"] == "T-1"
    assert captured["json"]["thread_id"] == "thr-abc"


# ---------------------------------------------------------------------------
# mark_read
# ---------------------------------------------------------------------------


def test_mark_read_hits_correct_path():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return httpx.Response(200, json={"ok": True})

    client = _client_with_handler(handler)
    asyncio.run(client.mark_read(123))
    asyncio.run(client.aclose())
    assert captured["path"] == "/messages/123/read"


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_sends_self_name_in_body():
    captured = {}

    def handler(request):
        import json
        captured["json"] = json.loads(request.read())
        return httpx.Response(200, json={
            "name": "lab-ovh", "enabled": True,
            "url": "http://lab-ovh:8787",
        })

    client = _client_with_handler(handler)
    asyncio.run(client.register(
        url="http://lab-ovh:8787",
        capabilities={"runtime": "claude", "can_claim_tasks": True},
    ))
    asyncio.run(client.aclose())
    assert captured["json"]["name"] == "lab-ovh"
    assert captured["json"]["url"] == "http://lab-ovh:8787"
    assert captured["json"]["capabilities"]["runtime"] == "claude"


# ---------------------------------------------------------------------------
# Auth / error handling
# ---------------------------------------------------------------------------


def test_401_raises_mesh_auth_error():
    def handler(request):
        return httpx.Response(401, json={"detail": "missing token"})

    client = _client_with_handler(handler)
    with pytest.raises(MeshAuthError, match="MESH_GATEWAY_TOKEN"):
        asyncio.run(client.list_peers())
    asyncio.run(client.aclose())


def test_403_raises_mesh_auth_error():
    def handler(request):
        return httpx.Response(403, json={"detail": "forbidden"})

    client = _client_with_handler(handler)
    with pytest.raises(MeshAuthError):
        asyncio.run(client.list_peers())
    asyncio.run(client.aclose())


def test_500_raises_mesh_gateway_error():
    def handler(request):
        return httpx.Response(500, text="internal error")

    client = _client_with_handler(handler)
    with pytest.raises(MeshGatewayError, match="500"):
        asyncio.run(client.list_peers())
    asyncio.run(client.aclose())


def test_request_error_wraps_as_mesh_gateway_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    client = _client_with_handler(handler)
    with pytest.raises(MeshGatewayError, match="request failed"):
        asyncio.run(client.list_peers())
    asyncio.run(client.aclose())


def test_non_json_response_raises():
    def handler(request):
        return httpx.Response(200, text="<html>not json</html>")

    client = _client_with_handler(handler)
    with pytest.raises(MeshGatewayError, match="non-JSON"):
        asyncio.run(client.list_peers())
    asyncio.run(client.aclose())


# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


def test_construction_falls_back_to_env_token(monkeypatch):
    monkeypatch.setenv("MESH_GATEWAY_TOKEN", "env-token-12345")
    client = MeshClient(node="lab-ovh", validate_self_name=False)
    assert client._token == "env-token-12345"


def test_construction_falls_back_to_env_url(monkeypatch):
    monkeypatch.setenv("MESH_GATEWAY_URL", "http://custom-gateway:9999")
    client = MeshClient(node="lab-ovh", token="t", validate_self_name=False)
    assert client._gateway_url == "http://custom-gateway:9999"


def test_construction_uses_default_url_when_no_env(monkeypatch):
    """#548: the fallback is the tailnet IP — the gateway binds no loopback."""
    monkeypatch.delenv("MESH_GATEWAY_URL", raising=False)
    client = MeshClient(node="lab-ovh", token="t", validate_self_name=False)
    assert client._gateway_url == "http://100.107.222.72:8788"


def test_self_name_alias_resolves_at_construction(monkeypatch):
    """Constructing a client with an alias resolves to canonical."""
    # Skip registry check (offline), but regex + alias should still resolve
    client = MeshClient(node="drop", validate_self_name=True, token="t")
    assert client.node == "droplet"


def test_async_context_manager_closes_client():
    handler = lambda request: httpx.Response(200, json={"peers": []})

    async def _go():
        async with MeshClient(node="lab-ovh", token="t", validate_self_name=False) as c:
            # Force a connection so _client is real
            transport = httpx.MockTransport(handler)
            c._client = httpx.AsyncClient(
                base_url=c._gateway_url, transport=transport,
                headers={"Authorization": f"Bearer t"},
            )
            await c.list_peers()
            assert c._client is not None and not c._client.is_closed
        # After __aexit__, client is closed
        assert c._client is None

    asyncio.run(_go())


def test_check_no_secret_catches_bare_gateway_token(monkeypatch):
    """A bare MESH_GATEWAY_TOKEN value pasted into content (WITHOUT the literal
    string 'MESH_GATEWAY_TOKEN' co-occurring) must be caught. The regex lookahead
    missed this — the common leak case (adversarial-sweep MED)."""
    tok = "Zk9xQ2vT" + "A1b2C3d4" * 6   # 56 chars, token-shaped, no lookahead trigger
    monkeypatch.setenv("MESH_GATEWAY_TOKEN", tok)
    with pytest.raises(MeshSecretLeakError):
        _check_no_secret(f"hey here's the gateway creds: {tok} thanks")


def test_check_no_secret_clean_when_token_not_present(monkeypatch):
    monkeypatch.setenv("MESH_GATEWAY_TOKEN", "Zk9xQ2vT" + "A1b2C3d4" * 6)
    _check_no_secret("a perfectly normal mesh message about the weather")  # no raise


def test_check_no_secret_ignores_short_or_unset_token(monkeypatch):
    # Empty/short env token must NOT make every message a false positive.
    monkeypatch.setenv("MESH_GATEWAY_TOKEN", "x")
    _check_no_secret("x marks the spot")  # no raise (len < 16 guard)

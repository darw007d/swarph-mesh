"""``MeshClient`` — async wrapper around the mesh-gateway HTTP API per
PLAN.md §5.

Replaces the hand-rolled curl-based code in:

* an inbox-drain script
* assorted ad-hoc curl invocations in session ritual

with a single typed surface so cross-Claude DM coordination doesn't
require re-discovering the gateway shape from scratch every time.

Public surface:

  client = MeshClient(node="my-cell", token=os.environ["MESH_GATEWAY_TOKEN"])
  peers = await client.list_peers()
  msgs = await client.fetch(unread_only=True)
  sent = await client.send(to="other-cell", kind="fyi", content="...")
  await client.mark_read(msg_id=123)
  await client.register(url="http://localhost:8787", capabilities={...})

Two structural invariants enforced:

1. **Recipient name validation.** Every ``send()`` calls
   ``swarph_shared.validate_node_name`` on the recipient — closes the
   Vector A (peer-onboarding chatter) + Vector B (human-prompt
   shorthand) framing-contagion classes documented in
   ``project_peer_name_canonical.md``.

2. **Mesh-secrets out-of-band guard.** Every ``send()`` runs a
   best-effort regex sniff on content for obvious credentials
   (``pypi-...``, ``sk-...``, ``gh[ops]_...``, ``eyJ...`` JWTs).
   Hits raise :class:`MeshSecretLeakError` BEFORE the POST. CLAUDE.md
   "Mesh secrets out-of-band only — NEVER ride /messages content
   fields" is the non-negotiable rule. Best-effort because regex
   misses are common — operators must still treat content fields as
   public. The guard catches the obvious cases.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional


# v0.5.1 (drop DM #722 + #728): client-side enum for mesh-gateway-accepted
# DM kinds. Gateway validates server-side too; this is fail-fast at the
# type-check + runtime layer so callers don't round-trip a known-bad value
# (e.g., kind="review_request" → 400). Same shape as the validate_caller
# pattern from swarph_shared. Add new kinds here when the gateway ships
# them (current accepted set per server.py mesh-gateway):
DM_KIND = Literal["fyi", "question", "answer", "status", "unblock"]

import httpx
from swarph_shared import validate_node_name

from swarph_mesh.exceptions import SwarphMeshError
from swarph_mesh.mesh_types import MeshMessage, MeshPeer


# TAILNET IP, NOT localhost (card #548; commander 2026-08-21). The mesh-gateway
# binds HOST=100.107.222.72 ONLY — localhost has never been bound, so this
# fallback failed as a bare "Connection refused" with no cause named.
# MESH_GATEWAY_URL (consulted at the call site) is the escape hatch for anyone
# outside this mesh, and optional inside it.
DEFAULT_GATEWAY_URL = "http://100.107.222.72:8788"
GATEWAY_TOKEN_ENV = "MESH_GATEWAY_TOKEN"
DEFAULT_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MeshGatewayError(SwarphMeshError):
    """Gateway returned non-2xx or otherwise non-conformant response."""


class MeshAuthError(MeshGatewayError):
    """Authentication failed — token missing or rejected (401/403)."""


class MeshSecretLeakError(SwarphMeshError):
    """Best-effort guard caught an apparent credential in DM content.

    Refuses to send. Operators must rotate the credential and resend
    without it.
    """


# ---------------------------------------------------------------------------
# Best-effort credential leak detector
# ---------------------------------------------------------------------------

# Patterns chosen to match observable shapes of credentials seen in
# session JSONLs in the field. Updates here are cheap; misses are
# expected (defense-in-depth, not strict validation).
_CRED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pypi token", re.compile(r"\bpypi-[A-Za-z0-9_-]{20,}")),
    ("anthropic api key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    # OpenAI ADMIN keys (highest privilege class — can create/delete keys,
    # manage org). Listed FIRST so it matches before the generic openai
    # pattern, giving the operator a properly scary error message.
    # Endpoint that mints these: POST /v1/organization/admin_api_keys.
    ("openai ADMIN key (org-level privilege)", re.compile(r"\bsk-admin-[A-Za-z0-9_-]{20,}")),
    # OpenAI key shapes: legacy ``sk-<48hex>`` + project keys
    # ``sk-proj-<long>`` (newer, longer alphabet incl. ``_-``). Earlier
    # regex ``sk-[A-Za-z0-9]{20,}`` missed sk-proj-... because of the
    # underscore + hyphen in the body. Issue #4 tightening.
    # NOTE: ``sk-admin-...`` is matched by the named pattern above so
    # this catch-all doesn't downgrade an admin key's error severity.
    ("openai api key", re.compile(r"\bsk-(?!admin-)(?:proj-)?[A-Za-z0-9_-]{20,}")),
    # xAI / Grok key — xai-<long>. Same shape as OpenAI's hyphen-prefix
    # pattern. Added in v0.5.1 alongside the Grok adapter shipped in
    # v0.5.0; would have matched the key the user pasted in chat earlier
    # this session if mesh send had been the destination.
    ("xai api key", re.compile(r"\bxai-[A-Za-z0-9_-]{40,}")),
    # DeepSeek api key — sk-<32hex> hex-only post the sk- prefix.
    # Already covered loosely by the openai pattern but listing
    # explicitly so the error message names the right provider.
    ("deepseek api key", re.compile(r"\bsk-[a-f0-9]{32}\b")),
    # Google Gemini / Cloud API keys — ``AIza<35-alnum>`` format.
    # Common shape across Google's API tier; would catch GEMINI_API_KEY
    # leaks. Issue #4 tightening (was missing).
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("github token", re.compile(r"\bgh[ops]_[A-Za-z0-9]{30,}")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws secret key", re.compile(r"\b[A-Za-z0-9/+=]{40}\b(?=.*aws)")),  # heuristic
    # JWT — three base64url segments separated by dots (stricter than the
    # generic shape so we don't false-fire on every blob with two dots).
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b")),
    # Mesh gateway tokens — long random base64url-shaped values stored in
    # MESH_GATEWAY_TOKEN env. Heuristic shape: 40+ chars of url-safe
    # base64 alphabet. False-positive rate higher than other patterns —
    # operator escape via skip_secret_check=True is the right hatch.
    ("possible mesh-gateway token", re.compile(r"\b[A-Za-z0-9_-]{48,}\b(?=.*MESH_GATEWAY_TOKEN)")),
]


def _check_no_secret(content: str) -> None:
    """Raise :class:`MeshSecretLeakError` if content looks like it
    carries a credential. Best-effort; expected to miss novel shapes.
    """
    # Exact-match the LIVE gateway token value. The regex pattern below uses a
    # `(?=.*MESH_GATEWAY_TOKEN)` lookahead, so it only fires when the literal
    # string 'MESH_GATEWAY_TOKEN' co-occurs — a bare token PASTE (the common
    # leak) sailed through. Comparing against the actual env value is exact
    # (zero false positives) and catches that case. (adversarial-sweep MED.)
    env_tok = os.environ.get(GATEWAY_TOKEN_ENV, "").strip()
    if env_tok and len(env_tok) >= 16 and env_tok in content:
        raise MeshSecretLeakError(
            f"refusing to send: content contains the live {GATEWAY_TOKEN_ENV} "
            "value. Mesh secrets out-of-band only — NEVER ride /messages content "
            "fields. (CLAUDE.md mesh-secrets rule.)"
        )
    for label, rx in _CRED_PATTERNS:
        if rx.search(content):
            raise MeshSecretLeakError(
                f"refusing to send: content matches {label} pattern. "
                "Mesh secrets out-of-band only — NEVER ride /messages "
                "content fields. Use ssh+tee, commander manual paste, "
                "or pre-redacted Hub records. (CLAUDE.md mesh-secrets rule.)"
            )


# ---------------------------------------------------------------------------
# MeshClient
# ---------------------------------------------------------------------------


class MeshClient:
    """Async client for the mesh-gateway HTTP API.

    Construct once per peer (or per cycle), reuse for the lifetime of
    the work. Backed by ``httpx.AsyncClient`` so connection-pooling
    happens for free across many ``send()`` calls.

    Use as an async context manager when you want explicit close:

        async with MeshClient(node="my-cell") as client:
            await client.send(to="other-cell", kind="fyi", content="...")

    Or instantiate normally + call ``aclose()`` when done.
    """

    def __init__(
        self,
        node: str,
        *,
        token: Optional[str] = None,
        gateway_url: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        validate_self_name: bool = True,
    ):
        """``token`` falls back to ``MESH_GATEWAY_TOKEN`` env. ``gateway_url``
        falls back to ``MESH_GATEWAY_URL`` env, then the tailnet IP
        (``http://100.107.222.72:8788`` — the gateway binds no loopback, #548).

        Set ``validate_self_name=False`` for test fixtures that need to
        instantiate a client with a mock peer name. Production callers
        should leave it on (caller's own ``node`` is the most common
        contagion target — see project_peer_name_canonical.md).
        """
        # Self-name validation — best-effort. If the gateway is offline
        # we still want construction to succeed; the validation is
        # registry-dependent so use strict=False to skip the registry
        # check when offline.
        if validate_self_name:
            try:
                node = validate_node_name(node, strict=False)
            except Exception:  # pragma: no cover — defensive
                pass

        self.node = node
        self._token = token if token is not None else os.environ.get(GATEWAY_TOKEN_ENV)
        self._gateway_url = (
            gateway_url
            or os.environ.get("MESH_GATEWAY_URL")
            or DEFAULT_GATEWAY_URL
        )
        self._timeout = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self._gateway_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> "MeshClient":
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal HTTP wrapper
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        client = self._ensure_client()
        try:
            resp = await client.request(method, path, params=params, json=json)
        except httpx.RequestError as exc:
            raise MeshGatewayError(
                f"mesh-gateway {method} {path} request failed: {exc}"
            ) from exc

        if resp.status_code == 401 or resp.status_code == 403:
            raise MeshAuthError(
                f"mesh-gateway {method} {path} returned {resp.status_code}: "
                f"check MESH_GATEWAY_TOKEN env"
            )
        if resp.status_code >= 400:
            body = resp.text[:500]
            raise MeshGatewayError(
                f"mesh-gateway {method} {path} returned {resp.status_code}: {body}"
            )

        # Gateway returns JSON for everything except /health
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise MeshGatewayError(
                f"mesh-gateway {method} {path} returned non-JSON: {resp.text[:200]}"
            ) from exc

    # ------------------------------------------------------------------
    # Peer endpoints
    # ------------------------------------------------------------------

    async def list_peers(self) -> list[MeshPeer]:
        """``GET /peers`` — return all registered peers."""
        payload = await self._request("GET", "/peers")
        rows = payload.get("peers", payload) if isinstance(payload, dict) else payload
        return [MeshPeer.model_validate(r) for r in rows]

    async def register(
        self,
        *,
        url: Optional[str] = None,
        capabilities: Optional[dict[str, Any]] = None,
    ) -> MeshPeer:
        """``POST /peers/register`` — register this peer with the gateway."""
        body: dict[str, Any] = {"name": self.node}
        if url is not None:
            body["url"] = url
        if capabilities is not None:
            body["capabilities"] = capabilities
        payload = await self._request("POST", "/peers/register", json=body)
        return MeshPeer.model_validate(payload)

    # ------------------------------------------------------------------
    # Message endpoints
    # ------------------------------------------------------------------

    async def fetch(
        self,
        *,
        to_node: Optional[str] = None,
        unread_only: bool = False,
        limit: Optional[int] = None,
    ) -> list[MeshMessage]:
        """``GET /messages`` — fetch own inbox.

        Defaults ``to_node`` to ``self.node``. Pass an explicit value
        only if you have a legitimate reason to peek at another peer's
        inbox (gateway access-controls this; most callers will get
        their own inbox only).

        v0.5.1 latent bug fix (discovered in Phase 5.6 daemon build):
        the mesh-gateway query parameter is ``?to=`` — NOT ``?to_node=``.
        The latter is silently ignored, returning ALL recent messages
        regardless of recipient. Pre-fix every ``MeshClient.fetch``
        caller saw outbound bleed-through unless they Python-side
        filtered. The kwarg keeps its descriptive name for the
        caller-facing API; only the wire-shape changes.
        """
        # Defense-in-depth: filter from_node!=self_node client-side too,
        # so any future gateway quirk that re-introduces outbound
        # bleed-through doesn't silently break consumers (same shape
        # as the daemon's filter).
        target = to_node or self.node
        params: dict[str, Any] = {"to": target}
        if unread_only:
            params["unread_only"] = "true"
        if limit is not None:
            params["limit"] = limit
        payload = await self._request("GET", "/messages", params=params)
        rows = payload.get("messages", payload) if isinstance(payload, dict) else payload
        # Client-side defensive filter — only if fetching own inbox
        # (i.e., target == self.node). When peeking at another peer's
        # inbox via explicit to_node= override, return whatever gateway
        # returns without filtering (caller asked for raw view).
        if target == self.node:
            rows = [r for r in rows if r.get("from_node") != self.node]
        return [MeshMessage.model_validate(r) for r in rows]

    async def send(
        self,
        *,
        to: str,
        kind: DM_KIND,
        content: str,
        related_task_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        skip_secret_check: bool = False,
    ) -> MeshMessage:
        """``POST /messages`` — send a DM.

        Validates ``to`` via ``swarph_shared.validate_node_name`` (closes
        Vector A + B contagion classes); raises ``ValueError`` /
        ``NotInRegistry`` on invalid recipient names BEFORE the POST.

        ``kind`` is a ``Literal`` enum (see ``DM_KIND`` above) — type
        checkers reject unknown values at lint time, and the runtime
        guard below catches anyone passing ``kind`` as ``str``-typed
        from a callsite that bypassed type-checking.

        Runs the best-effort credential leak detector on ``content``.
        Set ``skip_secret_check=True`` only when content has been
        inspected by the caller and the leak detector is producing a
        false positive (e.g. discussing a credential format in
        prose). False positives should be rare; treat them as a
        signal that the content is on a fence and rotate aggressively
        if you're not 100% sure.
        """
        # Runtime kind enum check — defense-in-depth against str-typed
        # callsites that slipped past type-checking. Same shape as
        # validate_caller fail-fast discipline.
        if kind not in {"fyi", "question", "answer", "status", "unblock"}:
            raise ValueError(
                f"MeshClient.send: kind={kind!r} is not a valid mesh-gateway "
                f"DM kind. Accepted: fyi / question / answer / status / unblock. "
                f"(Adding new kinds requires gateway-side support; check "
                f"server.py POST /messages handler.)"
            )

        # Recipient validation — FAIL-CLOSED (strict=True). A send is about to
        # hit the same gateway via POST /messages, so the registry's /peers is
        # reachable in the normal case (and a warm TTL cache covers transient
        # blips). strict=False used to fail OPEN: a /peers hiccup skipped the
        # registry check and let a regex-valid-but-unregistered name through,
        # POSTing the DM into the void (adversarial-sweep auth-bypass HIGH).
        # Pass this client's own gateway_url + token so the registry check hits
        # the SAME gateway we POST to (not the MESH_GATEWAY_URL env default).
        canonical_to = validate_node_name(
            to, strict=True, gateway_url=self._gateway_url, token=self._token,
        )

        # Best-effort credential leak guard. Operator can disable for
        # legitimate prose mentioning credential shapes.
        if not skip_secret_check:
            _check_no_secret(content)

        body: dict[str, Any] = {
            "from_node": self.node,
            "to_node": canonical_to,
            "kind": kind,
            "content": content,
        }
        if related_task_id is not None:
            body["related_task_id"] = related_task_id
        if thread_id is not None:
            body["thread_id"] = thread_id

        payload = await self._request("POST", "/messages", json=body)
        return MeshMessage.model_validate(payload)

    async def mark_read(self, msg_id: int) -> None:
        """``POST /messages/{msg_id}/read`` — flip ``read_at`` on a DM.

        Per the mesh DM drain convention, the cron-side
        watcher does NOT mark read; only the commander or the
        peer-in-session does. Adapter callers consuming inboxes
        should mark-read explicitly when they've actually processed
        the DM, not on fetch.
        """
        await self._request("POST", f"/messages/{msg_id}/read")

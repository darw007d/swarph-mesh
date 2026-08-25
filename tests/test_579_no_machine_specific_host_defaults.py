"""Guard: no machine-specific IP literal may ship inside swarph-mesh.

Written against the PROPERTY, not the syntax of the current bug. #546's finder
(`grep 'GATEWAY[A-Z_]* *= *"http'`) was disarmed by its own fix — the literal moved
into a fallback argument and the query went permanently blind. drop-on-meta-edge:
"IT FOUND THE BUG ONCE AND IS NOW PERMANENTLY BLIND TO IT."

So this sweeps for any CGNAT/RFC1918 address in shipped source — the property that
makes such a default wrong whatever shape it wears. Loopback is allowed: wrong on
this fleet, but harmless anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "swarph_mesh"

_MACHINE_SPECIFIC = re.compile(
    r"""\b(
          100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}
        | 10\.\d{1,3}\.\d{1,3}\.\d{1,3}
        | 172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}
        | 192\.168\.\d{1,3}\.\d{1,3}
    )\b""",
    re.VERBOSE,
)

_TEXT_SUFFIXES = {".py", ".md", ".default", ".service", ".timer", ".sh", ".toml", ".json"}


def _sweep(root: Path) -> list[str]:
    """Root is an ARGUMENT so the can-fail case can scan a synthetic bad tree."""
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            found = _MACHINE_SPECIFIC.search(line)
            if found:
                offenders.append(f"{path.name}:{lineno}: {found.group(0)}  |  {line.strip()[:90]}")
    return offenders


def test_no_machine_specific_ip_ships_in_the_package() -> None:
    offenders = _sweep(SRC)
    assert not offenders, (
        "Machine-specific address(es) shipped in swarph-mesh.\n"
        "A default that names one box expires the day that box is retired "
        "(#578/#579).\n\n" + "\n".join(offenders)
    )


def test_the_guard_actually_fires(tmp_path: Path) -> None:
    """CAN-FAIL: prove the sweep is not vacuously green."""
    (tmp_path / "bad.py").write_text('X = "http://100.107.' + '222.72:8788"\n')
    assert _sweep(tmp_path)


def test_loopback_is_deliberately_allowed(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text('X = "http://127.0.0.1:8788"\n')
    assert not _sweep(tmp_path)


def test_the_module_default_is_EMPTY_when_the_env_is_unset() -> None:
    """The property, OBSERVED — not set by the test that checks it.

    swarph-shared's sibling test was proven vacuous by can-fail: it monkeypatched
    DEFAULT_GATEWAY_URL to "" and then asserted the refusal, so re-introducing a
    host literal left it green. Reading the shipped constant is what cannot be
    disarmed that way.
    """
    from swarph_mesh import mesh_client

    assert mesh_client.DEFAULT_GATEWAY_URL == "", (
        f"swarph-mesh ships a gateway host default: "
        f"{mesh_client.DEFAULT_GATEWAY_URL!r} (#578/#579)"
    )


def test_client_refuses_rather_than_guessing_a_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The behaviour the constant's removal is FOR.

    Without this, deleting the literal would look identical to changing it — the
    package would resolve to "" and fail later with something unreadable.
    """
    from swarph_mesh.mesh_client import MeshClient

    monkeypatch.delenv("MESH_GATEWAY_URL", raising=False)
    with pytest.raises(ValueError, match="MESH_GATEWAY_URL is not set"):
        MeshClient(node="probe")


def test_client_uses_the_env_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarph_mesh.mesh_client import MeshClient

    monkeypatch.setenv("MESH_GATEWAY_URL", "http://gw.example:8788")
    assert MeshClient(node="probe")._gateway_url == "http://gw.example:8788"


def test_explicit_argument_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarph_mesh.mesh_client import MeshClient

    monkeypatch.setenv("MESH_GATEWAY_URL", "http://env.example:8788")
    c = MeshClient(node="probe", gateway_url="http://explicit.example:8788")
    assert c._gateway_url == "http://explicit.example:8788"

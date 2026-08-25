

# --- #578/#579: swarph-mesh ships NO default gateway host -------------------
#
# MeshClient now refuses to construct without one. Most of this suite is testing
# something else entirely (auth errors, response shapes, adapters) and merely
# needs A client, so it gets a dummy gateway here rather than 27 near-identical
# edits.
#
# THIS IS NOT MASKING THE PROPERTY. The refusal itself is covered by dedicated
# tests that DELETE the variable and assert the failure —
# test_579_no_machine_specific_host_defaults.py::
#   test_client_refuses_rather_than_guessing_a_host
#   test_client_uses_the_env_when_set
#   test_explicit_argument_still_wins
# and test_mesh_client.py::test_construction_refuses_when_no_env.
#
# The distinction matters: a fixture that silently supplied a REAL host would
# disarm the guard (#546's lesson — "it found the bug once and is now
# permanently blind to it"). A dummy host that no test asserts against does not.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _mesh_gateway_configured(monkeypatch):
    monkeypatch.setenv("MESH_GATEWAY_URL", "http://gateway.invalid:8788")

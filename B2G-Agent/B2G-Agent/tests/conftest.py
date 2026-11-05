from __future__ import annotations

import pytest
import socket
import os
import tempfile
from pathlib import Path


# Application modules create their default store at import time, before fixtures.
# Keep collection independent of model downloads and private user sessions.
_collection_root = Path(__file__).resolve().parents[1] / ".personal"
_collection_root.mkdir(exist_ok=True)
_collection_workspace = tempfile.TemporaryDirectory(prefix="pytest-collection-", dir=_collection_root)
os.environ["B2G_EMBEDDING_BACKEND"] = "hash_test_only"
os.environ["B2G_LLM_MODE"] = "offline"
os.environ["B2G_LLM_ENABLED"] = "false"
os.environ["B2G_PERSONAL_ROOT"] = _collection_workspace.name


def pytest_unconfigure(config):
    _collection_workspace.cleanup()


@pytest.fixture(autouse=True)
def isolate_external_simulators(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Unit tests never call a live LLM or workstation-specific simulator."""

    monkeypatch.setenv("B2G_LLM_MODE", "offline")
    monkeypatch.setenv("B2G_LLM_ENABLED", "false")
    monkeypatch.setenv("B2G_EMBEDDING_BACKEND", "hash_test_only")
    monkeypatch.delenv("B2G_DATA_ROOT", raising=False)
    monkeypatch.delenv("B2G_EVENT_JOURNAL", raising=False)
    monkeypatch.setenv("B2G_JOURNAL_ROOT", str(tmp_path / "events"))
    monkeypatch.setenv("B2G_ONLINE_MEMORY_ENABLED", "false")

    def deny_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline tests")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def local_only(original):
        def guarded(sock, address):
            # Windows asyncio builds its internal socketpair over loopback.
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return original(sock, address)
            return deny_network()
        return guarded

    monkeypatch.setattr(socket.socket, "connect", local_only(original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", local_only(original_connect_ex))
    monkeypatch.setattr(socket, "create_connection", deny_network)
    from openai.resources.chat.completions import Completions, AsyncCompletions
    monkeypatch.setattr(Completions, "create", deny_network)
    monkeypatch.setattr(AsyncCompletions, "create", deny_network)

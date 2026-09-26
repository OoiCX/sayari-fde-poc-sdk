"""Block real network access for every test and share the pipeline fixtures with both suites."""

import socket

import pytest

from tests.pipeline_support import (
    audit_case as audit_case,
)
from tests.pipeline_support import (
    cache as cache,
)
from tests.pipeline_support import (
    findings as findings,
)
from tests.pipeline_support import (
    forbid_external_stages as forbid_external_stages,
)
from tests.pipeline_support import (
    graph as graph,
)
from tests.pipeline_support import (
    settings as settings,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Real network access is forbidden in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sdk-cache-dir",
        help="Read-only S0 corpus to validate; omitted runs use synthetic temporary files.",
    )

    parser.addoption(
        "--sdk-cache-root",
        help="Read-only reference tree; inventories and measures all response caches.",
    )

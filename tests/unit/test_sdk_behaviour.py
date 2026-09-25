"""Execute the five S0 probes against the installed SDK, never a live endpoint."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
import sayari.base_client
import sayari.core.http_client
from sayari import Sayari
from sayari.core.api_error import ApiError
from sayari.core.request_options import RequestOptions
from sayari.supply_chain import UpstreamTradeTraversalResponse

Handler = Callable[[httpx.Request], httpx.Response]


@contextmanager
def mocked_sdk(
    monkeypatch: pytest.MonkeyPatch, data_handler: Handler, auth_handler: Handler
) -> Iterator[Sayari]:
    # SDK 0.1.43 builds a second httpx client for automatic OAuth. Capture that route separately,
    # because injecting the data client alone does not cover it.
    with ExitStack() as stack:
        data_client = stack.enter_context(
            httpx.Client(transport=httpx.MockTransport(data_handler), trust_env=False)
        )

        def internal_client(**kwargs: Any) -> httpx.Client:
            return stack.enter_context(
                httpx.Client(transport=httpx.MockTransport(auth_handler), trust_env=False, **kwargs)
            )

        monkeypatch.setattr(sayari.base_client, "httpx", SimpleNamespace(Client=internal_client))
        yield Sayari(
            base_url="https://sdk-conformance.invalid",
            client_id="",
            client_secret="",
            httpx_client=data_client,
        )


def token_response(request: httpx.Request) -> httpx.Response:
    assert request.url.host == "sdk-conformance.invalid"
    assert request.url.path == "/oauth/token"
    return httpx.Response(
        200,
        json={"access_token": uuid4().hex, "expires_in": 3600, "token_type": "Bearer"},
    )


def retry_attempts(monkeypatch: pytest.MonkeyPatch, status: int, max_retries: int | None) -> int:
    data_paths: list[str] = []
    auth_paths: list[str] = []
    waits: list[float] = []

    def data(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "sdk-conformance.invalid"
        data_paths.append(request.url.path)
        return httpx.Response(status, headers={"Retry-After": "1"}, json={})

    def auth(request: httpx.Request) -> httpx.Response:
        auth_paths.append(request.url.path)
        return token_response(request)

    with monkeypatch.context() as probe:
        # Keep the SDK's own retry decisions and timeout calculation; suppress only the wall-clock
        # waiting, and record every delay it asks for.
        probe.setattr(
            sayari.core.http_client,
            "time",
            SimpleNamespace(sleep=waits.append, time=time.time),
        )
        with mocked_sdk(probe, data, auth) as client, pytest.raises(ApiError) as error:
            if max_retries is None:
                client.supply_chain.upstream_trade_traversal("probe")
            else:
                client.supply_chain.upstream_trade_traversal(
                    "probe", request_options=RequestOptions(max_retries=max_retries)
                )
    assert error.value.status_code == status
    assert auth_paths == ["/oauth/token"]
    assert data_paths == ["/v1/supply_chain/upstream/probe"] * len(data_paths)
    assert waits == [1.0] * (len(data_paths) - 1)
    return len(data_paths)


def test_pinned_sdk_retry_header_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Record the pinned SDK's retry-header limits and the headers it fails to parse.
    module = sayari.core.http_client
    monkeypatch.setattr(module, "random", lambda: 0.0)
    assert module.MAX_RETRY_DELAY_SECONDS_FROM_HEADER == 30
    assert module.MAX_RETRY_DELAY_SECONDS == 10
    long_block = httpx.Response(429, headers={"Retry-After": "60"})
    milliseconds = httpx.Response(429, headers={"retry-after-ms": "1500"})
    assert module._parse_retry_after(long_block.headers) == 60.0
    assert module._parse_retry_after(milliseconds.headers) is None
    assert module._retry_timeout(long_block, retries=2) == 2.0
    assert module._retry_timeout(milliseconds, retries=2) == 2.0
    assert module._retry_timeout(long_block, retries=10) == 10.0


def test_retry_attempts_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each retry setting produces the expected number of mocked attempts.
    print("\nstatus | default | N=1 | N=2 | N=3 | N=5")
    for status in (429, 500, 408):
        attempts_row = [
            retry_attempts(monkeypatch, status, max_retries) for max_retries in (None, 1, 2, 3, 5)
        ]
        print(f"{status} | " + " | ".join(map(str, attempts_row)))
        assert attempts_row == [1, 1, 1, 2, 4]
    print("Retry-After: 1 -> every measured retry requested sleep(1.0).")


def test_offline_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing the SDK alone performs no network request.
    attempted: list[str] = []

    def blocked(request: httpx.Request) -> httpx.Response:
        attempted.append(request.url.path)
        raise AssertionError("Construction performed I/O")

    with mocked_sdk(monkeypatch, blocked, blocked) as client:
        assert client is not None
        assert attempted == []
    print("\nConstruction: success; HTTP requests=0; credential arguments empty.")


@pytest.mark.parametrize("entity_id", ["part/part", "part+part"])
def test_entity_path_escaping(monkeypatch: pytest.MonkeyPatch, entity_id: str) -> None:
    # The SDK puts IDs into the path unescaped, which is why the adapter validates IDs itself.
    paths: list[tuple[str, bytes]] = []

    def data(request: httpx.Request) -> httpx.Response:
        paths.append((request.url.path, request.url.raw_path))
        return httpx.Response(408, json={})

    with mocked_sdk(monkeypatch, data, token_response) as client, pytest.raises(ApiError):
        # Exercise the endpoint the adapter actually calls, so this records the escaping behaviour
        # that reaches production rather than a sibling endpoint's.
        client.entity.entity_summary(entity_id)
    expected = f"/v1/entity_summary/{entity_id}"
    assert paths == [(expected, expected.encode("ascii"))]
    print(f"\nPath id={entity_id!r}: path={paths[0][0]!r}; raw_path={paths[0][1]!r}")


def test_upstream_laxness() -> None:
    # The pinned SDK coerces raw coverage values that the adapter has to reject.
    parsed = UpstreamTradeTraversalResponse.model_validate(
        {
            "filters": {},
            "data": {"paths": [], "entities": {}},
            "partial_results": "false",
            "explored_count": True,
        }
    )
    assert parsed.partial_results is False
    assert type(parsed.explored_count) is int
    assert parsed.explored_count == 1
    print('\nLaxness: partial_results="false" -> False (bool); explored_count=true -> 1 (int).')


def test_auth_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK sends token requests through a different client from its data requests.
    data_paths: list[str] = []
    internal_paths: list[str] = []

    def data(request: httpx.Request) -> httpx.Response:
        data_paths.append(request.url.path)
        return httpx.Response(408, json={})

    def auth(request: httpx.Request) -> httpx.Response:
        internal_paths.append(request.url.path)
        return token_response(request)

    with mocked_sdk(monkeypatch, data, auth) as client, pytest.raises(ApiError):
        client.supply_chain.upstream_trade_traversal("probe")
    assert internal_paths == ["/oauth/token"]
    assert data_paths == ["/v1/supply_chain/upstream/probe"]
    assert "/oauth/token" not in data_paths
    print(
        "\nAuth visibility: injected transport auth=0 data=1; "
        "SDK-created transport auth=1 data=0; OAuth path=/oauth/token."
    )


def test_socket_guard_is_active() -> None:
    # The suite-wide guard blocks every real socket access.
    with socket.socket() as sock:
        calls: list[Callable[[], object]] = [
            lambda: sock.connect(("sdk-conformance.invalid", 443)),
            lambda: sock.connect_ex(("sdk-conformance.invalid", 443)),
            lambda: socket.create_connection(("sdk-conformance.invalid", 443)),
            lambda: socket.getaddrinfo("sdk-conformance.invalid", 443),
        ]
        for call in calls:
            with pytest.raises(AssertionError, match="Real network access is forbidden in tests"):
                call()

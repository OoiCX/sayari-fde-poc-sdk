"""P1 acceptance through the real SDK and mocked socket transport only."""

from __future__ import annotations

import ast
import importlib
import json
import ssl
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

import sayari_poc.sayari_sdk as boundary
import sayari_poc.transport as audited
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.models import InputEntity
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import (
    BudgetExceeded,
    OfflineCacheMiss,
    RateLimitPacer,
    SayariAuthError,
    SayariError,
    SayariNotFound,
    SayariRateLimitError,
    SayariValidationError,
)

# Reuse the synthetic payload builders instead of keeping a second copy here.
from tests.sdk_support import candidate, profile

TOKEN = "synthetic-in-process-access-value"
Handler = Callable[[httpx.Request], httpx.Response]


def upstream_payload() -> dict[str, Any]:
    return {
        "filters": {},
        "partial_results": False,
        "explored_count": 1,
        "data": {
            "entities": {
                "node": {
                    "id": "node",
                    "type": "company",
                    "label": "Synthetic node",
                    "countries": ["USA", "SGP"],
                    "risk_factors": [],
                    "translated_label": "Synthetic translation",
                },
            },
            "paths": [
                {
                    "source_entity_id": "root",
                    "path": [
                        {
                            "entity_id": "node",
                            "tier": 3,
                            "components": [
                                {
                                    "hs_code": "1234",
                                    "arrival_countries": ["SGP", "USA"],
                                    "departure_countries": ["CHN"],
                                    "min_date": "2020-01-01",
                                },
                            ],
                        },
                    ],
                }
            ],
        },
    }


@dataclass
class Clock:
    now: float = 0.0
    waits: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now

    def pacer(self) -> RateLimitPacer:
        return RateLimitPacer(sleep=self.sleep, monotonic=self.monotonic)


@dataclass
class Probe:
    orphan_count: int = 0
    orphan_requests: int = 0
    providers: list[Any] = field(default_factory=list)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        sayari_api_base="https://p1.invalid",
        sayari_client_id="synthetic-client-identifier",
        sayari_client_secret="synthetic-client-secret",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def cache(settings: Settings) -> ResponseCache:
    return ResponseCache(settings.cache_dir)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Probe:
    evidence = Probe()
    sdk_base = importlib.import_module("sayari.base_client")
    original_provider = importlib.import_module(
        "sayari.core.oauth_token_provider"
    ).OAuthTokenProvider

    def unused(request: httpx.Request) -> httpx.Response:
        evidence.orphan_requests += 1
        raise AssertionError("The SDK's orphan client must be unused")

    def orphan(**kwargs: Any) -> httpx.Client:
        evidence.orphan_count += 1
        return httpx.Client(transport=httpx.MockTransport(unused), trust_env=False, **kwargs)

    def provider(**kwargs: Any) -> Any:
        result = original_provider(**kwargs)
        evidence.providers.append(result)
        return result

    monkeypatch.setattr(sdk_base, "httpx", SimpleNamespace(Client=orphan))
    # Preserve the public buffer constant that the facade reads before constructing the provider.
    monkeypatch.setattr(
        provider, "BUFFER_IN_MINUTES", original_provider.BUFFER_IN_MINUTES, raising=False
    )
    monkeypatch.setattr(boundary, "OAuthTokenProvider", provider)
    return evidence


def responding(data: Handler, *, expires_in: int = 3600) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "p1.invalid"
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200, json={"access_token": TOKEN, "expires_in": expires_in, "token_type": "Bearer"}
            )
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        return data(request)

    return handle


def seed(cache: ResponseCache, path: str, payload: object) -> bytes:
    content = json.dumps(payload, ensure_ascii=False, indent=2).encode()
    cache.put(cache.key(httpx.Request("GET", "https://p1.invalid" + path)), content)
    return content


@pytest.mark.parametrize("failure_status", [200, 201])
def test_explicit_upstream_failure_is_not_cached_but_success_bytes_are(
    settings: Settings, cache: ResponseCache, probe: Probe, failure_status: int
) -> None:
    # A traversal that reports failure is never cached, even when the HTTP status is a success.
    success = json.dumps(upstream_payload(), indent=3).encode()
    failure = {**upstream_payload(), "success": False}
    responses = iter(
        [httpx.Response(failure_status, json=failure), httpx.Response(200, content=success)]
    )
    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(responding(lambda request: next(responses))),
        pacer=Clock().pacer(),
    ) as client:
        failed = client.upstream("root", max_depth=2, limit=10)
        assert failed.status == "error" and failed.error_type == "ValidationError"
        assert failed.explored_count == 1 and failed.partial_results is False
        assert list(cache.cache_dir.glob("*.json")) == []
        successful = client.upstream("root", max_depth=2, limit=10)
        assert successful.status == "assessed"
        assert client.audit.data_http_attempts == 2
        assert [path.read_bytes() for path in cache.cache_dir.glob("*.json")] == [success]
        assert client.upstream("root", max_depth=2, limit=10) == successful
        assert client.audit.data_http_attempts == 2 and client.cache_hits == 1
        assert probe.orphan_requests == 0


def test_baseclient_override_auth_count_and_orphan_unused(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # SDK endpoint calls and token requests both go through the audited client override.
    seen: list[str] = []

    def data(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert set(request.extensions["timeout"].values()) == {60.0}
        return httpx.Response(200, json=profile())

    with SayariClient(
        settings, cache, transport=httpx.MockTransport(responding(data)), pacer=Clock().pacer()
    ) as client:
        assert client._replay_sdk._client_wrapper._token == probe.providers[0].get_token
        assert probe.orphan_count == 1
        assert client.get_entity("root").entity_id == "root"
        assert client._live_sdk is not None
        assert client._live_sdk._client_wrapper._token == probe.providers[1].get_token
        assert probe.orphan_count == 2  # Exactly one per SDK instance.
        assert probe.orphan_requests == 0
        assert seen == ["/v1/entity_summary/root"]
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 1
        assert client.cache_lookups == 1
        assert [a.kind for a in client.audit.attempts] == ["auth", "data"]
        assert [a.status for a in client.audit.attempts] == [200, 200]
        assert not hasattr(client._replay_sdk, "screen_csv")
    assert client._http.is_closed


@pytest.mark.parametrize("offline", [True, False])
def test_cached_bytes_need_no_credentials_no_env_no_ca_or_socket(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    monkeypatch: pytest.MonkeyPatch,
    offline: bool,
) -> None:
    # Replaying from the cache needs no credentials, proxy, CA bundle or socket.
    settings = settings.model_copy(
        update={
            "sayari_client_id": None,
            "sayari_client_secret": None,
            "sayari_ca_bundle": "absent-ca-file.pem",
        }
    )
    seed(cache, "/v1/entity_summary/root", profile())

    def no_ca(**kwargs: object) -> ssl.SSLContext:
        raise AssertionError("A cache hit must not read a CA file")

    monkeypatch.setattr(audited, "ssl", SimpleNamespace(create_default_context=no_ca))
    with SayariClient(settings, cache, offline=offline) as client:
        assert client.get_entity("root").entity_id == "root"
        assert client.cache_hits == client.cache_lookups == 1
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 0
        if offline:
            with pytest.raises(OfflineCacheMiss):
                client.get_entity("absent")
        assert probe.orphan_count == 1
        assert probe.orphan_requests == 0


def test_custom_ca_reaches_auth_and_data_transport(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A custom CA bundle reaches the real transport for both auth and data requests.
    settings = settings.model_copy(update={"sayari_ca_bundle": "test-corporate-ca.pem"})
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    loaded: list[object] = []
    seen: list[str] = []

    def create_context(*, cafile: object) -> ssl.SSLContext:
        loaded.append(cafile)
        return context

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return responding(lambda req: httpx.Response(200, json=profile()))(request)

    def socket_transport(**kwargs: object) -> httpx.MockTransport:
        assert kwargs == {"verify": context, "retries": 0, "trust_env": False}
        return httpx.MockTransport(handle)

    monkeypatch.setattr(audited, "ssl", SimpleNamespace(create_default_context=create_context))
    monkeypatch.setattr(httpx, "HTTPTransport", socket_transport)
    with SayariClient(settings, cache, pacer=Clock().pacer()) as client:
        assert loaded == []
        client.get_entity("root")
        assert loaded == ["test-corporate-ca.pem"]
        assert seen == ["/oauth/token", "/v1/entity_summary/root"]
        assert probe.orphan_requests == 0


@pytest.mark.parametrize(
    ("field_name", "value", "partial", "explored"),
    [("explored_count", True, False, None), ("partial_results", "false", False, 1)],
)
@pytest.mark.parametrize("cached", [False, True])
def test_raw_coverage_fails_closed_before_sdk_coercion(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    field_name: str,
    value: object,
    partial: bool,
    explored: int | None,
    cached: bool,
) -> None:
    # Invalid raw coverage values cannot be made valid by the SDK's type coercion.
    payload = upstream_payload()
    payload[field_name] = value
    if cached:
        seed(cache, "/v1/supply_chain/upstream/root?max_depth=3&limit=500", payload)
    with SayariClient(
        settings,
        cache,
        offline=cached,
        transport=httpx.MockTransport(responding(lambda req: httpx.Response(200, json=payload))),
        pacer=Clock().pacer(),
    ) as client:
        result = client.upstream("root")
        assert result.status == "error"
        assert result.error_type == "ValidationError"
        assert result.partial_results is partial
        assert result.explored_count == explored
        assert result.entities == {} and result.paths == []
        assert client.cache_hits == int(cached)
        assert client.cache_lookups == 1


@pytest.fixture
def sdk_clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(
        importlib.import_module("sayari.core.http_client"),
        "time",
        SimpleNamespace(sleep=clock.sleep, time=time.time),
    )
    monkeypatch.setattr(
        importlib.import_module("sayari.core.oauth_token_provider"),
        "dt",
        SimpleNamespace(
            datetime=SimpleNamespace(
                now=lambda: datetime(2026, 1, 1) + timedelta(seconds=clock.now)
            ),
            timedelta=timedelta,
        ),
    )
    return clock


def test_budget_stops_sdk_retry_storm_before_third_data_attempt(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    sdk_clock: Clock,
) -> None:
    # The transport's call ceiling stops retries before a data send the budget does not allow.
    settings = settings.model_copy(update={"call_budget": 2})
    attempts: list[str] = []

    def storm(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        return httpx.Response(429, headers={"Retry-After": "1"}, json={})

    with SayariClient(
        settings, cache, transport=httpx.MockTransport(responding(storm)), pacer=sdk_clock.pacer()
    ) as client:
        with pytest.raises(BudgetExceeded):
            client.get_entity("root")
        assert len(attempts) == client.calls_made == 2
        assert client.audit.auth_http_attempts == 1
        assert client.cache_lookups == 1 and client.cache_hits == 0
        assert [a.status for a in client.audit.attempts] == [200, 429, 429]
        assert all(a.retry_after for a in client.audit.attempts[1:])
        seed(cache, "/v1/entity_summary/cached", profile("cached"))
        assert client.get_entity("cached").entity_id == "cached"
        with pytest.raises(BudgetExceeded):
            client.get_entity("still-refused")
        assert len(attempts) == 2
    assert not list(settings.cache_dir.glob("*.tmp"))


def test_auth_requests_are_neither_charged_to_nor_gated_by_the_data_budget(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    sdk_clock: Clock,
) -> None:
    """Token requests have their own counter and the ceiling never refuses one.

    OAuth must never be charged to the budget, and the guard in AuditedTransport._send is
    therefore `not auth and ...`. Dropping `not auth` still passed the whole suite, so this pins the
    surviving half: after the budget is exhausted a token refresh must reach the transport and
    increment only the auth counter, while the data attempt behind it is refused.
    """
    settings = settings.model_copy(update={"call_budget": 1})
    data_requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        data_requests.append(request.url.path)
        return httpx.Response(200, json=profile(request.url.path.rsplit("/", 1)[-1]))

    # Advance past a valid token expiry so the SDK renews it after the data budget is exhausted.
    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(responding(handler)),
        pacer=sdk_clock.pacer(),
    ) as client:
        assert client.get_entity("root").entity_id == "root"
        assert client.audit.auth_http_attempts == 1 and client.calls_made == 1
        sdk_clock.now += 3600
        with pytest.raises(BudgetExceeded):
            client.get_entity("second")
        # The token request ran and was counted; only the data attempt behind it was refused.
        assert client.audit.auth_http_attempts == 2
        assert client.calls_made == 1
        assert data_requests == ["/v1/entity_summary/root"]
        assert [attempt.kind for attempt in client.audit.attempts] == ["auth", "data", "auth"]


def test_retries_share_one_lookup_and_replay_is_identical(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    sdk_clock: Clock,
) -> None:
    # All retries share one logical cache lookup, and the replayed bytes match the live response.
    statuses = iter([429, 500, 408, 200])
    raw = json.dumps(profile(), ensure_ascii=False, indent=3).encode()

    def retry_then_success(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(
            status, content=raw if status == 200 else b"{}", headers={"Retry-After": "1"}
        )

    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(responding(retry_then_success)),
        pacer=sdk_clock.pacer(),
    ) as client:
        live = client.get_entity("root")
        assert client.calls_made == 4 and client.cache_lookups == 1
        assert client.audit.auth_http_attempts == 1
        assert sdk_clock.waits.count(1.0) == 3
        assert client.get_entity("root") == live
        assert client.calls_made == 4 and client.cache_hits == 1
        assert client.cache_lookups == 2
    assert next(settings.cache_dir.glob("*.json")).read_bytes() == raw
    with SayariClient(settings, cache, offline=True) as replay:
        assert replay.get_entity("root").model_dump_json() == live.model_dump_json()
        assert replay.calls_made == replay.audit.auth_http_attempts == 0


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, SayariAuthError),
        (403, SayariAuthError),
        (404, SayariNotFound),
        (429, SayariRateLimitError),
        (500, SayariError),
        (408, SayariError),
    ],
)
def test_error_responses_are_safe_and_never_cached(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    sdk_clock: Clock,
    status: int,
    error: type[SayariError],
) -> None:
    # An HTTP failure raises a safe error that hides the response body, and nothing is cached.
    with SayariClient(
        settings,
        cache,
        pacer=sdk_clock.pacer(),
        transport=httpx.MockTransport(
            responding(
                lambda req: httpx.Response(
                    status,
                    json={
                        "status": status,
                        "message": ["response-must-not-escape"],
                        "success": False,
                    },
                    headers={"Retry-After": "1"},
                )
            )
        ),
    ) as client:
        with pytest.raises(error) as caught:
            client.get_entity("root")
        assert "response-must-not-escape" not in "".join(traceback.format_exception(caught.value))
        assert client.calls_made == (4 if status in (429, 500, 408) else 1)
    assert not list(settings.cache_dir.glob("*.json"))


@pytest.mark.parametrize("content", [b"{broken", b"[]", b"null", b"42", b"{}"])
def test_corrupt_cache_and_sdk_invalid_cache_never_fall_through_to_http(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    content: bytes,
) -> None:
    # A corrupt or invalid cache entry raises an error; it never falls back to a live call.
    key = cache.key(httpx.Request("GET", "https://p1.invalid/v1/entity_summary/root"))
    settings.cache_dir.mkdir()
    (settings.cache_dir / f"{key}.json").write_bytes(content)
    with SayariClient(settings, cache) as client:
        with pytest.raises(SayariValidationError):
            client.get_entity("root")
        assert client.calls_made == client.audit.auth_http_attempts == 0
    assert (settings.cache_dir / f"{key}.json").read_bytes() == content


@pytest.mark.parametrize("source", ["client_id", "secret", "token"])
@pytest.mark.parametrize("escaped", [False, True])
def test_credential_echo_including_unicode_escapes_is_rejected_before_write(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    source: str,
    escaped: bool,
) -> None:
    # A response that echoes a credential, even in escaped form, is rejected before caching.
    values = {
        "client_id": settings.sayari_client_id,
        "secret": settings.sayari_client_secret.get_secret_value()
        if settings.sayari_client_secret
        else "",
        "token": TOKEN,
    }
    value = values[source]
    assert value is not None
    encoded = "".join(f"\\u{ord(char):04x}" for char in value) if escaped else value
    raw = ('{"echo":"' + encoded + '"}').encode()
    with SayariClient(
        settings,
        cache,
        pacer=Clock().pacer(),
        transport=httpx.MockTransport(responding(lambda req: httpx.Response(200, content=raw))),
    ) as client:
        with pytest.raises(SayariError, match="authentication credential") as caught:
            client.get_entity("root")
        assert value not in str(caught.value)
    assert not list(settings.cache_dir.glob("*.json"))


def test_outbound_credential_in_query_is_refused_before_socket(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # A query that contains a credential is refused before any authentication or data request.
    assert settings.sayari_client_id is not None
    with SayariClient(settings, cache) as client:
        result = client.resolve(
            InputEntity(
                name=settings.sayari_client_id,
                address=None,
                country=None,
                sheet="list_3",
                row_number=2,
            )
        )
        assert result.status == "error" and result.error_type == "SayariError"
        assert client.calls_made == client.audit.auth_http_attempts == 0
    assert not list(settings.cache_dir.glob("*.json"))


def test_tier_pacing_and_coverage_are_independent_of_tier_policy(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pacing depends on the endpoint's tier, but the tier never changes how coverage is read.
    clock = Clock()
    pacer = clock.pacer()
    assert pacer.wait("/v1/supply_chain/upstream/root") == 2
    assert pacer.wait("/v1/supply_chain/upstream/other") == 2
    assert clock.waits == [pytest.approx(10 / 15)]
    for path in ("/v1/entity_summary/root", "/v1/resolution", "/oauth/token", "/v1/ontology/risk"):
        assert pacer.wait(path) == 1
    assert clock.waits[1:] == [pytest.approx(60 / 200)] * 3
    monkeypatch.setattr(RateLimitPacer, "tier", staticmethod(lambda path: 1))
    payload = upstream_payload()
    payload["explored_count"] = True
    seed(cache, "/v1/supply_chain/upstream/root?max_depth=3&limit=500", payload)
    with SayariClient(settings, cache, offline=True) as client:
        assert client.upstream("root").status == "error"
        assert client.cache_hits == 1


def test_sdk_types_are_confined_to_adapter_in_production() -> None:
    # In production code, only the SDK facade imports the SDK.
    root = Path(__file__).resolve().parents[2] / "src/sayari_poc"
    imported: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.py")):
        modules: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        imported[path.name] = [m for m in modules if m == "sayari" or m.startswith("sayari.")]
    assert imported["sayari_sdk.py"]
    assert all(not imports for name, imports in imported.items() if name != "sayari_sdk.py")


def test_token_rotation_uses_sdk_provider_and_same_audit_transport(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    sdk_clock: Clock,
) -> None:
    # When the SDK rotates its token, the new token request still uses the audited transport.
    with SayariClient(
        settings,
        cache,
        pacer=sdk_clock.pacer(),
        transport=httpx.MockTransport(
            responding(
                lambda req: httpx.Response(200, json=profile(req.url.path.rsplit("/", 1)[1])),
                expires_in=3600,
            )
        ),
    ) as client:
        client.get_entity("first")
        sdk_clock.now += 3600
        client.get_entity("second")
        assert client.audit.auth_http_attempts == client.calls_made == 2
        assert probe.orphan_count == 2 and probe.orphan_requests == 0


def test_redirect_hops_are_budgeted_and_cache_original_sdk_request(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # Each redirect hop uses budget, and the response is cached under the original request.
    def redirect(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/root"):
            return httpx.Response(307, headers={"Location": "/redirected-profile"})
        return httpx.Response(200, json=profile())

    with SayariClient(
        settings, cache, pacer=Clock().pacer(), transport=httpx.MockTransport(responding(redirect))
    ) as client:
        client.get_entity("root")
        assert client.calls_made == 2 and client.cache_lookups == 1
        assert [a.status for a in client.audit.attempts] == [200, 307, 200]
    with SayariClient(settings, cache, offline=True) as client:
        assert client.get_entity("root").entity_id == "root"
        assert client.cache_hits == 1 and client.calls_made == 0


def test_cross_origin_redirect_is_refused_before_second_send(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # A cross-origin redirect fails before credentials can be sent anywhere else.
    with SayariClient(
        settings,
        cache,
        pacer=Clock().pacer(),
        transport=httpx.MockTransport(
            responding(
                lambda req: httpx.Response(
                    307, headers={"Location": "https://other.invalid/entity"}
                )
            )
        ),
    ) as client:
        with pytest.raises(SayariError, match="Cross-origin"):
            client.get_entity("root")
        assert client.calls_made == 1
    assert not list(settings.cache_dir.glob("*.json"))


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("method", ["entity", "resolution", "upstream"])
def test_process_control_exceptions_always_propagate(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    failure: type[BaseException],
    method: str,
) -> None:
    # The adapter never turns KeyboardInterrupt or SystemExit into a row error.
    def stop(request: httpx.Request) -> httpx.Response:
        raise failure()

    with SayariClient(
        settings, cache, pacer=Clock().pacer(), transport=httpx.MockTransport(responding(stop))
    ) as client:
        with pytest.raises(failure):
            if method == "entity":
                client.get_entity("root")
            elif method == "upstream":
                client.upstream("root")
            else:
                client.resolve(
                    InputEntity(
                        name="Synthetic", address=None, country=None, sheet="list_3", row_number=2
                    )
                )


def test_failed_transport_counts_and_returns_safe_error(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # A failed send is counted, and the transport exception's details never leak.
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private transport details", request=request)

    with SayariClient(
        settings, cache, pacer=Clock().pacer(), transport=httpx.MockTransport(responding(fail))
    ) as client:
        with pytest.raises(SayariError) as caught:
            client.get_entity("root")
        assert "private transport details" not in "".join(traceback.format_exception(caught.value))
        assert client.calls_made == 1 and client.audit.attempts[-1].status is None


@pytest.mark.parametrize("entity_id", ["", "a/b", "a+b", "../x", "a%2Fb", "a?x", "a#x"])
def test_unsafe_ids_fail_before_authentication(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    entity_id: str,
) -> None:
    # An unsafe canonical ID is rejected before any auth or data attempt.
    with SayariClient(settings, cache) as client:
        with pytest.raises(SayariValidationError):
            client.get_entity(entity_id)
        assert client.upstream(entity_id).status == "error"
        assert client.calls_made == client.audit.auth_http_attempts == 0


@pytest.mark.parametrize(
    ("strength", "status"), [("strong", "resolved"), ("weak", "weak"), ("unknown", "error")]
)
def test_resolution_order_scores_and_fail_closed_strength(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    strength: str,
    status: str,
) -> None:
    # Resolution keeps source order and the relevance score, and never infers match strength.
    payload = {
        "fields": {},
        "data": [candidate("first", strength), candidate("second", score=999.0)],
    }
    with SayariClient(
        settings,
        cache,
        pacer=Clock().pacer(),
        transport=httpx.MockTransport(responding(lambda req: httpx.Response(200, json=payload))),
    ) as client:
        result = client.resolve(
            InputEntity(name="Synthetic", address=None, country=None, sheet="list_3", row_number=2)
        )
        assert result.status == status
        if status != "error":
            assert [c.entity_id for c in result.candidates] == ["first", "second"]
            assert result.score == 130.5 and result.entity_id == "first"
        else:
            assert result.entity_id is None and result.error_type == "ValidationError"


def test_upstream_preserves_path_order_metadata_and_isolates_later_failure(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # Traversal keeps the path evidence intact, and a later failed call does not affect it.
    payload = upstream_payload()
    payload["partial_results"] = True
    payload["data"]["paths"] *= 2
    seed(cache, "/v1/supply_chain/upstream/root?max_depth=3&limit=500", payload)
    with SayariClient(settings, cache, offline=True) as client:
        result = client.upstream("root")
        assert result.status == "partial" and result.partial_results is True
        assert result.explored_count == 1
        assert [p.path_index for p in result.paths] == [0, 1]
        assert result.paths[0].hops[0].tier == 3
        assert result.paths[0].hops[0].components[0].arrival_countries == ["SGP", "USA"]
        assert result.entities["node"].translated_label == "Synthetic translation"
        bad_id = client.upstream("bad/id")
        assert bad_id.status == "error" and bad_id.partial_results is False
        assert bad_id.explored_count is None
        malformed = upstream_payload()
        del malformed["filters"]
        malformed["partial_results"] = True
        seed(cache, "/v1/supply_chain/upstream/broken?max_depth=3&limit=500", malformed)
        failed = client.upstream("broken")
        assert failed.status == "error" and failed.partial_results is True
        assert failed.explored_count == 1 and failed.paths == [] and failed.entities == {}


def test_refresh_replaces_corrupt_cache_but_offline_refresh_is_rejected(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # Refresh replaces a corrupt cache entry, but it cannot be combined with offline mode.
    with pytest.raises(ValueError, match="incompatible"):
        SayariClient(settings, cache, offline=True, refresh=True)
    key = cache.key(httpx.Request("GET", "https://p1.invalid/v1/entity_summary/root"))
    settings.cache_dir.mkdir()
    (settings.cache_dir / f"{key}.json").write_bytes(b"broken")
    with SayariClient(
        settings,
        cache,
        refresh=True,
        pacer=Clock().pacer(),
        transport=httpx.MockTransport(responding(lambda req: httpx.Response(200, json=profile()))),
    ) as client:
        assert client.get_entity("root").entity_id == "root"
        assert client.cache_lookups == client.cache_hits == 0
        assert client.calls_made == 1
    assert json.loads((settings.cache_dir / f"{key}.json").read_bytes()) == profile()


@pytest.mark.parametrize(("partial", "expected"), [(False, "no_data"), (True, "partial")])
def test_empty_success_is_not_confused_with_error(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    partial: bool,
    expected: str,
) -> None:
    # A successful but empty traversal keeps its own coverage outcome and is never an error.
    payload = {
        "filters": {},
        "partial_results": partial,
        "explored_count": 0,
        "data": {"entities": {}, "paths": []},
    }
    seed(cache, "/v1/supply_chain/upstream/root?max_depth=3&limit=500", payload)
    with SayariClient(settings, cache, offline=True) as client:
        result = client.upstream("root")
        assert result.status == expected and result.error_type is None
        assert result.partial_results is partial and result.explored_count == 0


def test_http_library_logging_cannot_expose_supplier_query(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # HTTP-library debug logs never leak supplier queries or OAuth traffic.
    caplog.set_level("DEBUG")
    with SayariClient(
        settings,
        cache,
        pacer=Clock().pacer(),
        transport=httpx.MockTransport(
            responding(lambda req: httpx.Response(200, json={"fields": {}, "data": []}))
        ),
    ) as client:
        result = client.resolve(
            InputEntity(
                name="Synthetic-Private-Supplier",
                address=None,
                country=None,
                sheet="list_3",
                row_number=2,
            )
        )
        assert result.status == "no_match"
    assert "Synthetic-Private-Supplier" not in caplog.text
    assert "/oauth/token" not in caplog.text


def test_coverage_validation_follows_original_upstream_request_through_redirect(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # A redirect cannot skip the raw coverage checks that apply to the original endpoint.
    payload = upstream_payload()
    payload["explored_count"] = True

    def redirect(request: httpx.Request) -> httpx.Response:
        if "/upstream/" in request.url.path:
            return httpx.Response(307, headers={"Location": "/redirected-traversal"})
        return httpx.Response(200, json=payload)

    with SayariClient(
        settings, cache, pacer=Clock().pacer(), transport=httpx.MockTransport(responding(redirect))
    ) as client:
        result = client.upstream("root")
        assert result.status == "error" and result.error_type == "ValidationError"
        assert result.explored_count is None and client.calls_made == 2


def test_path_validation_error_retains_raw_coverage(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # Invalid paths keep the coverage metadata for diagnosis, but no partial path list.
    payload = upstream_payload()
    payload["partial_results"] = True
    payload["data"]["paths"][0]["path"][0]["entity_id"] = "absent"
    seed(cache, "/v1/supply_chain/upstream/root?max_depth=3&limit=500", payload)
    with SayariClient(settings, cache, offline=True) as client:
        result = client.upstream("root")
        assert result.status == "error" and result.error_type == "ValidationError"
        assert result.partial_results is True and result.explored_count == 1
        assert result.entities == {} and result.paths == []


def test_urlencoded_credential_with_spaces_is_blocked_before_auth(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
) -> None:
    # A credential containing spaces is still caught after URL encoding, before any auth request.
    settings = settings.model_copy(update={"sayari_client_id": "synthetic spaced + value"})
    with SayariClient(settings, cache) as client:
        result = client.resolve(
            InputEntity(
                name="synthetic spaced + value",
                address=None,
                country=None,
                sheet="list_3",
                row_number=2,
            )
        )
        assert result.status == "error" and result.error_type == "SayariError"
        assert client.calls_made == client.audit.auth_http_attempts == 0


# The API origin guard is a security boundary: credentials, plain HTTP or a path-bearing base URL
# must be refused before any client or socket exists.
@pytest.mark.parametrize(
    "base",
    [
        "http://p1.invalid",
        "https://synthetic-user:synthetic-pass@p1.invalid",
        "https://p1.invalid/v1",
        "https://p1.invalid?region=eu",
        "https://p1.invalid#fragment",
        "https:///v1",
    ],
)
def test_unsafe_api_origin_is_refused_before_any_request(
    settings: Settings, cache: ResponseCache, base: str
) -> None:
    # An invalid API origin fails when the client is built, before any request is sent.
    sent: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(500)

    unsafe = settings.model_copy(update={"sayari_api_base": base})
    with pytest.raises(ValueError, match="HTTPS origin without credentials"):
        SayariClient(unsafe, cache, transport=httpx.MockTransport(record), pacer=Clock().pacer())
    assert sent == []


# A successful auth response without a usable token must stop the run with a safe auth error,
# counted as an auth attempt, before any data request is sent.
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b"{}",
        b'{"access_token": ""}',
        b'{"access_token": "   "}',
        b'{"access_token": 7}',
        b'{"access_token": "synthetic-token"}',
        b'{"access_token": "synthetic-token", "expires_in": "invalid", "token_type": "Bearer"}',
        b'{"access_token": "synthetic-token", "expires_in": 3600}',
        b'{"access_token": "synthetic-token", "expires_in": 3600, "token_type": null}',
    ],
)
def test_malformed_auth_response_stops_before_any_data_request(
    settings: Settings, cache: ResponseCache, probe: Probe, body: bytes
) -> None:
    # A malformed token is attempted once across later calls, with no data requests or caching.
    data_paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})
        data_paths.append(request.url.path)
        return httpx.Response(200, json=profile())

    with SayariClient(
        settings, cache, transport=httpx.MockTransport(handle), pacer=Clock().pacer()
    ) as client:
        with pytest.raises(SayariAuthError, match="Malformed Sayari authentication response"):
            client.get_entity("root")
        for entity_id in ("later", "last"):
            with pytest.raises(SayariAuthError):
                client.get_entity(entity_id)
        assert data_paths == []
        assert client.audit.data_http_attempts == 0
        assert client.audit.auth_http_attempts == 1
        assert client.audit.pacing_waits == 0
    assert not list(settings.cache_dir.glob("*.json"))


@pytest.mark.parametrize("status", [302, 400, 401, 403, 429, 500])
def test_unsuccessful_token_response_latches_before_later_attempts(
    settings: Settings, cache: ResponseCache, probe: Probe, status: int
) -> None:
    # Every non-success token response stops later sign-ins before transport, pacing or counters.
    paths: list[str] = []

    def reject(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(status, content=b"synthetic-private-auth-detail")

    with SayariClient(
        settings, cache, refresh=True, transport=httpx.MockTransport(reject), pacer=Clock().pacer()
    ) as client:
        for entity_id in ("first", "second", "third"):
            with pytest.raises(SayariAuthError) as caught:
                client.get_entity(entity_id)
            assert "synthetic-private-auth-detail" not in str(caught.value)
        assert paths == ["/oauth/token"]
        assert client.audit.auth_http_attempts == 1
        assert client.audit.data_http_attempts == client.audit.pacing_waits == 0
    assert not list(settings.cache_dir.glob("*.json"))


def test_missing_token_credentials_never_send_or_pace(
    settings: Settings, cache: ResponseCache, probe: Probe
) -> None:
    # Cache misses without credentials fail safely on every call without a sign-in attempt.
    configured = settings.model_copy(
        update={"sayari_client_id": None, "sayari_client_secret": None}
    )
    paths: list[str] = []

    def unexpected(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        raise AssertionError("Missing credentials must not reach the send boundary")

    with SayariClient(
        configured, cache, transport=httpx.MockTransport(unexpected), pacer=Clock().pacer()
    ) as client:
        for entity_id in ("first", "second", "third"):
            with pytest.raises(SayariAuthError):
                client.get_entity(entity_id)
        assert paths == []
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 0
        assert client.audit.pacing_waits == 0


@pytest.mark.parametrize("expires_in", [10**100, 0, -1, 120])
def test_unusable_token_expiry_latches_before_data_requests(
    settings: Settings, cache: ResponseCache, probe: Probe, expires_in: int
) -> None:
    # Unrepresentable or already-buffered expiry fails once without data requests or cached bytes.
    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(
            responding(lambda req: httpx.Response(200, json=profile()), expires_in=expires_in)
        ),
        pacer=Clock().pacer(),
    ) as client:
        with pytest.raises(SayariAuthError, match="^Malformed Sayari authentication response$"):
            client.get_entity("first")
        for entity_id in ("second", "third"):
            with pytest.raises(SayariAuthError):
                client.get_entity(entity_id)
        assert client.audit.auth_http_attempts == 1
        assert client.audit.data_http_attempts == client.audit.pacing_waits == 0
    assert not list(settings.cache_dir.glob("*"))


def test_documented_token_expiry_reuses_one_token_for_three_calls(
    settings: Settings, cache: ResponseCache, probe: Probe
) -> None:
    # The documented one-day token stays valid across distinct uncached SDK data calls.
    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(
            responding(
                lambda req: httpx.Response(200, json=profile(req.url.path.rsplit("/", 1)[1])),
                expires_in=86400,
            )
        ),
        pacer=Clock().pacer(),
    ) as client:
        for entity_id in ("first", "second", "third"):
            assert client.get_entity(entity_id).entity_id == entity_id
        assert client.audit.auth_http_attempts == 1
        assert client.audit.data_http_attempts == 3


@pytest.mark.parametrize(
    ("partial", "has_upstream", "expected"),
    [(False, False, "no_data"), (True, False, "partial"), (False, True, "assessed")],
)
def test_root_entity_does_not_establish_upstream_coverage(
    settings: Settings,
    cache: ResponseCache,
    probe: Probe,
    partial: bool,
    has_upstream: bool,
    expected: str,
) -> None:
    # Retain the queried root while deriving coverage from other entities and the partial flag.
    payload = upstream_payload()
    entities = payload["data"]["entities"]
    root = {**entities["node"], "id": "root"}
    payload["data"]["entities"] = {"root": root, **(entities if has_upstream else {})}
    payload["data"]["paths"] = []
    payload["partial_results"] = partial
    seed(cache, "/v1/supply_chain/upstream/root?max_depth=3&limit=500", payload)
    with SayariClient(settings, cache, offline=True) as client:
        result = client.upstream("root")
    assert result.status == expected and result.error_type is None
    assert set(result.entities) == ({"root", "node"} if has_upstream else {"root"})
    assert result.entities["root"].label == root["label"]
    assert result.entities["root"].countries == root["countries"]
    assert result.paths == [] and result.partial_results is partial


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5, "120", None])
def test_transport_rejects_invalid_minimum_token_lifetime(
    settings: Settings, cache: ResponseCache, minimum: object
) -> None:
    # Reject nonpositive and noninteger lifetime configuration, including bool's int subclass.
    with pytest.raises(ValueError, match="^Minimum token lifetime must be a positive integer$"):
        audited.AuditedTransport(settings, cache, min_token_lifetime_seconds=cast(int, minimum))


def test_facade_passes_sdk_expiry_buffer_to_transport(
    settings: Settings, cache: ResponseCache, probe: Probe
) -> None:
    # Transport validation uses the SDK's public buffer without importing SDK types itself.
    provider = importlib.import_module("sayari.core.oauth_token_provider").OAuthTokenProvider
    with SayariClient(settings, cache, offline=True) as client:
        assert client._transport.min_token_lifetime_seconds == provider.BUFFER_IN_MINUTES * 60

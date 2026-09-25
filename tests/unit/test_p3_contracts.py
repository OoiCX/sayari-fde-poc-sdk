"""P3 boundary regressions: honest errors, lifecycle, telemetry and pure rendering."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import duckdb
import httpx
import pytest

from sayari_poc import pipeline, report
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.models import Findings, UpstreamResult
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import BudgetExceeded, RateLimitPacer, SayariError
from tests.ontology_support import pipeline_ontology as pipeline_ontology
from tests.p3_support import _prepare
from tests.unit.test_p2_stages import client_for, profile, upstream_payload

pytestmark = pytest.mark.usefixtures("pipeline_ontology")


@pytest.mark.parametrize("partial,explored", [(False, None), (True, 77)])
def test_error_metadata_is_never_read_or_rendered(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
    explored: int | None,
) -> None:
    # Coverage fields on an error result are never read, counted or rendered as coverage.
    reads: list[str] = []

    class DiagnosticOnly(UpstreamResult):
        def __getattribute__(self, name: str) -> Any:
            if name in {"partial_results", "explored_count"}:
                reads.append(name)
                raise AssertionError("Error coverage metadata must not be read")
            return super().__getattribute__(name)

    configured, _ = _prepare(settings, tmp_path, {"list_3": [("Input 1", "root")]})
    failure = DiagnosticOnly(
        supplier_id="root",
        entities={},
        paths=[],
        status="error",
        partial_results=partial,
        explored_count=explored,
        error_type="SayariRateLimitError",
    )
    monkeypatch.setattr(pipeline, "fetch_upstreams", lambda *args: {"root": failure})
    output = tmp_path / "output"
    findings = pipeline.run_pipeline(configured, offline=True, output_dir=output)
    row = findings.suppliers[0]
    assert row["coverage_status"] == "error"
    assert row["upstream_error_type"] == "SayariRateLimitError"
    assert row["upstream_entity_count"] is None
    assert row["upstream_partial_results"] is None
    html = (output / "report.html").read_text(encoding="utf-8")
    # The supplier section is now followed by the flagged-entity list. Ending the slice any later
    # would silently widen this check beyond the supplier section.
    supplier_section = html.split('<section id="suppliers"', 1)[1].split(
        '<section id="flagged-entities"', 1
    )[0]
    assert "upstream entities were found" not in supplier_section
    assert "partial_results:" not in supplier_section
    assert "SayariRateLimitError" in supplier_section
    assert reads == []


@pytest.mark.parametrize("tier", ["2", 2.0, True, 2])
def test_raw_tier_is_never_coerced(tmp_path: Path, tier: object) -> None:
    # A tier that is not a real integer fails before the SDK can coerce it into looking valid.
    payload = upstream_payload()
    payload["data"]["paths"][0]["path"][0]["tier"] = tier
    client, cache = client_for(tmp_path, payload)
    with client:
        result = client.upstream("root")
        if type(tier) is int:
            assert result.status == "assessed"
            assert result.error_type is None
            assert result.paths[0].hops[0].tier == tier
        else:
            assert result.status == "error"
            assert result.error_type == "ValidationError"
            assert result.entities == {} and result.paths == []
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 0
        assert client.audit.cache_hits == client.audit.cache_lookups == 1
        assert len(cache.keys) == 1


class ClosingTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.closed = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError("Offline cache replay must not send")

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "resolve_entities",
        "fetch_profiles",
        "fetch_upstreams",
        "build_warehouse",
        "render_report",
    ],
)
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_pipeline_closes_owned_transport_on_success_and_failure(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
    error: type[BaseException],
) -> None:
    # The pipeline closes the transport it owns exactly once, whether the run succeeds or fails.
    configured, _ = _prepare(settings, tmp_path, {"list_3": [("Input 1", "root")]})
    transport = ClosingTransport()
    original = SayariClient

    def make_client(*args: Any, **kwargs: Any) -> SayariClient:
        return original(*args, **kwargs, transport=transport)

    monkeypatch.setattr(pipeline, "SayariClient", make_client)
    if failure is not None:
        monkeypatch.setattr(pipeline, failure, Mock(side_effect=error("synthetic")))
        with pytest.raises(error):
            pipeline.run_pipeline(configured, offline=True, output_dir=tmp_path / "output")
    else:
        pipeline.run_pipeline(configured, offline=True, output_dir=tmp_path / "output")
    assert transport.closed == 1


@dataclass
class Clock:
    now: float = 0.0
    waits: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture
def telemetry_findings(settings: Settings) -> Findings:
    return Findings(
        generated_at="synthetic",
        suppliers=[],
        manifest={
            "sheets": [],
            "limit": None,
            "input_entities": 0,
            "hub_max_countries": settings.hub_max_countries,
            "max_upstream_depth": settings.max_upstream_depth,
            "upstream_limit": settings.upstream_limit,
            "taxonomy": {},
        },
    )


@pytest.mark.parametrize(
    "statuses,max_retries,budget,attempts,retries,wait_seconds",
    [
        ([429, 200], 5, 10, 2, 1, 0.6),
        ([503, 503, 503, 503], 5, 10, 4, 3, 1.2),
        ([503], 2, 10, 1, 0, 0.3),
        ([429, 429], 5, 2, 2, 1, 0.6),
    ],
)
def test_manifest_counts_actual_retries_and_pacing_only(
    settings: Settings,
    cache: ResponseCache,
    telemetry_findings: Findings,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[int],
    max_retries: int,
    budget: int,
    attempts: int,
    retries: int,
    wait_seconds: float,
) -> None:
    # The manifest's retry and pacing totals match the transport attempts that actually happened.
    findings = telemetry_findings
    settings.sdk_max_retries = max_retries
    settings.call_budget = budget
    responses = iter(statuses)
    clock = Clock()
    monkeypatch.setattr("sayari.core.http_client.time.sleep", lambda _: None)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        status = next(responses)
        return httpx.Response(status, json=profile() if status == 200 else {})

    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(respond),
        pacer=RateLimitPacer(clock.sleep, lambda: clock.now),
    ) as client:
        if statuses[-1] == 200:
            client.get_entity("root")
        else:
            with pytest.raises(BudgetExceeded if budget == 2 else SayariError):
                client.get_entity("root")
        api = cast(
            dict[str, Any],
            pipeline._run_manifest(
                findings,
                client,
                settings,
                offline=False,
                refresh=False,
            ),
        )["api"]
        assert api["data_http_attempts"] == attempts
        assert api["auth_http_attempts"] == 1
        assert api["retry_counts"] == {"auth": 0, "data": retries}
        assert api["pacing_waits"] == attempts
        assert api["pacing_wait_seconds"] == wait_seconds
        assert client.audit.pacing_wait_seconds == sum(clock.waits)
        assert len(client.audit.attempts) == attempts + 1
        assert api["cache_lookups"] == 1
        assert api["cache_hits"] == 0
        assert "pacing_wait_seconds" not in findings.manifest
        assert "retry_counts" not in findings.manifest


@pytest.mark.parametrize("redirect", [False, True])
def test_redirects_and_later_logical_calls_are_not_retries(
    settings: Settings, cache: ResponseCache, telemetry_findings: Findings, redirect: bool
) -> None:
    # Neither a redirect nor a later separate call is counted as a retry.
    findings = telemetry_findings
    settings.sdk_max_retries = 2
    clock = Clock()
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        attempts += 1
        if attempts == 1:
            return (
                httpx.Response(302, headers={"Location": "/redirected"})
                if redirect
                else httpx.Response(503, json={})
            )
        return httpx.Response(200, json=profile())

    with SayariClient(
        settings,
        cache,
        transport=httpx.MockTransport(respond),
        pacer=RateLimitPacer(clock.sleep, lambda: clock.now),
    ) as client:
        if not redirect:
            with pytest.raises(SayariError):
                client.get_entity("root")
        client.get_entity("root")
        api = cast(
            dict[str, Any],
            pipeline._run_manifest(
                findings,
                client,
                settings,
                offline=False,
                refresh=False,
            ),
        )["api"]
        assert api["retry_counts"] == {"auth": 0, "data": 0}
        assert api["data_http_attempts"] == 2
        assert api["auth_http_attempts"] == 1
        assert api["cache_lookups"] == (1 if redirect else 2)
        assert api["pacing_waits"] == 2
        assert api["pacing_wait_seconds"] == 0.6


def test_report_cannot_open_database_or_cache(
    findings: Findings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The report imports no storage or API module, and rendering never opens the warehouse or cache.
    tree = ast.parse(Path(report.__file__).read_text(encoding="utf-8"))
    imports = {
        name
        for node in ast.walk(tree)
        for name in (
            [node.module]
            if isinstance(node, ast.ImportFrom)
            else [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else []
        )
    }
    forbidden = {
        "duckdb",
        "sqlite3",
        "httpx",
        "sayari_poc.cache",
        "sayari_poc.analysis",
        "sayari_poc.sayari_sdk",
        "sayari_poc.transport",
    }
    assert imports.isdisjoint(forbidden)
    guard = Mock(side_effect=AssertionError("Rendering must not open evidence storage"))
    monkeypatch.setattr(duckdb, "connect", guard)
    monkeypatch.setattr(ResponseCache, "__init__", guard)
    monkeypatch.setattr(ResponseCache, "get", guard)
    monkeypatch.setattr(ResponseCache, "put", guard)
    report.render_report(findings, tmp_path / "report.html")
    guard.assert_not_called()

"""Source ontology and audited SDK acquisition contracts, using synthetic evidence."""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.risk_taxonomy import OntologySnapshotError, load_ontology
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import BudgetExceeded, RateLimitPacer, SayariValidationError


def payload() -> dict[str, object]:
    return {
        "filters": {},
        "total_count": 1,
        "data": [
            {
                "id": "psa_misleading_name",
                "label": "Published name",
                "description": "Published definition",
                "categories": ["synthetic"],
                "level": "high",
                "risk_type": "network",
                "doc": "",
                "code": 1,
                "risk_viz": "network",
                "visible": True,
                "enabled": True,
                "type": "integer",
                "tg": True,
                "do_not_render_metadata": [],
            }
        ],
    }


def snapshot(tmp_path: Path, data: dict[str, object] | None = None) -> Path:
    path = tmp_path / "ontology.json"
    raw = json.dumps(payload() if data is None else data).encode()
    path.write_bytes(raw)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-09-21T00:00:00+00:00",
                "filters": {},
                "sha256": hashlib.sha256(raw).hexdigest(),
                "sdk_version": "0.1.43",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_snapshot_exact_source_fields_and_no_prefix_inference(tmp_path: Path) -> None:
    # Ontology fields are loaded exactly as published, and nothing is inferred from an ID's prefix.
    ontology = load_ontology(snapshot(tmp_path))
    item = ontology.factors["psa_misleading_name"]
    assert (item.label, item.description, item.level, item.risk_type) == (
        "Published name",
        "Published definition",
        "high",
        "network",
    )
    assert "invented_high_sanctioned" not in ontology.factors
    assert ontology.parameters()["selected_levels"] == ["critical", "high"]


@pytest.mark.parametrize(
    "damage",
    ["missing", "metadata", "tampered", "filtered", "partial", "invalid_level", "duplicate"],
)
def test_snapshot_fails_loudly(tmp_path: Path, damage: str) -> None:
    # A missing, damaged or incomplete snapshot fails loudly instead of quietly classifying risk.
    data = payload()
    if damage == "filtered":
        data["filters"] = {"enabled": True}
    if damage == "partial":
        data["total_count"] = 2
    if damage == "invalid_level":
        data["data"] = [{**payload()["data"][0], "level": "invented"}]  # type: ignore[index]
    if damage == "duplicate":
        data["data"] = payload()["data"] * 2  # type: ignore[operator]
        data["total_count"] = 2
    path = snapshot(tmp_path, data)
    if damage == "missing":
        path.unlink()
    elif damage == "metadata":
        path.with_suffix(".metadata.json").unlink()
    elif damage == "tampered":
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(OntologySnapshotError, match="restore the committed snapshot"):
        load_ontology(path)


def test_ontology_uses_sdk_audited_budget_pacing_and_raw_cache(tmp_path: Path) -> None:
    # Ontology calls go through the SDK, count against the audited budget and cache raw bytes.
    seen: list[httpx.Request] = []
    raw = json.dumps(payload(), indent=3).encode()

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        assert request.url.path == "/v1/ontology/risk_factors"
        assert not request.url.query
        return httpx.Response(200, content=raw)

    settings = Settings(
        _env_file=None,
        sayari_client_id="synthetic-id",
        sayari_client_secret="synthetic-secret",
        call_budget=3,
        sdk_max_retries=0,
    )
    cache = ResponseCache(tmp_path / "cache")
    with SayariClient(
        settings, cache, refresh=True, transport=httpx.MockTransport(handle)
    ) as client:
        for _ in range(3):
            assert client.get_risk_factors().total_count == 1
        with pytest.raises(BudgetExceeded):
            client.get_risk_factors()
        assert client.audit.data_http_attempts == 3
        assert client.audit.auth_http_attempts == 1
        assert all(attempt.tier == 1 for attempt in client.audit.attempts)
    assert raw in [path.read_bytes() for path in cache.cache_dir.glob("*.json")]
    with SayariClient(settings, cache, offline=True) as replay:
        assert replay.get_risk_factors().data[0].label == "Published name"
        assert replay.audit.data_http_attempts == replay.audit.auth_http_attempts == 0
        assert replay.audit.cache_hits == 1
    assert len(seen) == 4
    assert RateLimitPacer.tier("/v1/ontology/risk_factors") == 1


def test_sdk_parse_failure_retains_raw_ontology_for_offline_validation(tmp_path: Path) -> None:
    # When SDK parsing fails, the raw ontology bytes are still cached for later offline replay.
    data = payload()
    cast(list[dict[str, Any]], data["data"])[0].pop("do_not_render_metadata")
    raw = json.dumps(data).encode()
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )
        return httpx.Response(200, content=raw)

    settings = Settings(
        _env_file=None,
        sayari_client_id="synthetic-id",
        sayari_client_secret="synthetic-secret",
        call_budget=1,
        sdk_max_retries=0,
    )
    cache = ResponseCache(tmp_path / "cache")
    with SayariClient(
        settings, cache, refresh=True, transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(SayariValidationError, match="Malformed Sayari response"):
            client.get_risk_factors()
        assert client.audit.data_http_attempts == 1
    assert seen == ["/oauth/token", "/v1/ontology/risk_factors"]
    cached = list(cache.cache_dir.glob("*.json"))
    assert len(cached) == 1 and cached[0].read_bytes() == raw
    path = snapshot(tmp_path, data)
    assert path.read_bytes() == cached[0].read_bytes()
    assert load_ontology(path).factors["psa_misleading_name"].level == "high"


def test_missing_snapshot_fails_before_client_or_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without the ontology snapshot the run stops before it fetches or writes anything.
    from unittest.mock import Mock

    from sayari_poc import pipeline, risk_taxonomy

    missing = tmp_path / "deleted.json"
    monkeypatch.setattr(risk_taxonomy, "SNAPSHOT_PATH", missing)
    monkeypatch.setattr(pipeline, "_ingest", lambda *args: {})
    client = Mock(side_effect=AssertionError("No client may be constructed"))
    monkeypatch.setattr(pipeline, "SayariClient", client)
    output = tmp_path / "output"
    with pytest.raises(OntologySnapshotError, match="No reclassification or network fallback"):
        pipeline.run_pipeline(Settings(_env_file=None), offline=True, output_dir=output)
    client.assert_not_called()
    assert not output.exists()


def test_unresolved_inventory_includes_unranked_upstream_factors(tmp_path: Path) -> None:
    # Unknown factors are still disclosed when the entities that carry them are not ranked.
    from sayari_poc.findings import assemble_findings
    from sayari_poc.models import UpstreamEntity, UpstreamResult

    ontology = load_ontology(snapshot(tmp_path))
    upstream = UpstreamResult(
        supplier_id="synthetic",
        status="assessed",
        partial_results=False,
        explored_count=1,
        entities={
            "node": UpstreamEntity(
                entity_id="node",
                countries=[],
                country_count=0,
                risk_factors=["psa_misleading_name", "unknown_high", "unknown_high"],
            )
        },
    )
    findings = assemble_findings(
        generated_at="synthetic",
        ingested={},
        inputs=[],
        resolved=[],
        profiles={},
        upstream={"synthetic": upstream},
        psa_records=[],
        ranked=[],
        convergence={},
        coverage={},
        headline_portfolio=None,
        limit=None,
        settings=Settings(_env_file=None),
        ontology=ontology,
    )
    taxonomy = findings.manifest["taxonomy"]
    assert isinstance(taxonomy, dict)
    assert taxonomy["observed_factor_count"] == 2
    assert taxonomy["resolved_factor_count"] == taxonomy["unresolved_factor_count"] == 1
    assert taxonomy["unresolved_factors"] == ["unknown_high"]
    assert taxonomy["missing_definition_count"] == 1
    assert list(findings.ontology) == ["psa_misleading_name"]
    assert findings.ontology["psa_misleading_name"].risk_type == "network"

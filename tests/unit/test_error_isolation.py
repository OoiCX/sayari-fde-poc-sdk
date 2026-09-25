"""Task 13 proves existing row isolation without adding catch sites."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import duckdb
import httpx
import pytest
from openpyxl import Workbook

from sayari_poc import pipeline
from sayari_poc.analysis import UnclassifiedRiskFactors
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.transport import AuditedTransport, OfflineCacheMiss
from tests.ontology_support import pipeline_ontology as pipeline_ontology
from tests.p3_support import _node, _upstream, entity_payload, resolution_payload

pytestmark = pytest.mark.usefixtures("pipeline_ontology")


@pytest.mark.parametrize("malformation", ["label", "countries", "factors", "blank", "source"])
def test_malformed_supplier_evidence_preserves_every_other_supplier(
    audit_case: tuple[Settings, ResponseCache],
    tmp_path: Path,
    malformation: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Malformed evidence fails only its own supplier; every other supplier's results survive.
    configured, cache = audit_case
    shared = _node("z-shared", 1, ["sanctioned_synthetic"])
    extra = _node("a-extra", 1, [])
    for supplier_id in ("a", "b", "c"):
        _upstream(cache, configured, supplier_id, {"z-shared": shared, "a-extra": extra})
    before = pipeline.run_pipeline(configured, offline=True, output_dir=tmp_path / "before")
    bad = {**shared}
    paths: list[dict[str, Any]] = []
    if malformation == "label":
        bad["label"] = "conflicting-sensitive-label"
    elif malformation == "countries":
        bad["countries"] = ["SGP", "USA"]
    elif malformation == "factors":
        bad["risk_factors"] = ["conflicting_sensitive_factor"]
    elif malformation == "blank":
        bad["risk_factors"] = [" "]
    else:
        paths = [
            {
                "source_entity_id": "wrong",
                "path": [{"tier": 2, "entity_id": "z-shared", "components": []}],
            }
        ]
    # A rejected supplier must not publish earlier nodes into the comparison set.
    _upstream(cache, configured, "a", {"z-shared": shared})
    _upstream(
        cache,
        configured,
        "b",
        {"a-extra": {**extra, "label": "rejected-sensitive-label"}, "z-shared": bad},
        partial=True,
        paths=paths,
    )
    output = tmp_path / "after"
    findings = pipeline.run_pipeline(configured, offline=True, output_dir=output)
    first, failed, later = findings.suppliers
    assert len(findings.suppliers) == 3
    assert first["coverage_status"] == later["coverage_status"] == "assessed"
    assert later == before.suppliers[2]
    assert first == {**before.suppliers[0], "upstream_entity_count": 1}
    assert failed["resolution_status"] == "resolved"
    assert failed["profile_status"] == "available"
    assert failed["coverage_status"] == "error"
    assert failed["upstream_entity_count"] is None
    assert failed["upstream_error_type"] == "ValidationError"
    assert findings.coverage["list_3"] == {
        "assessed": 2,
        "partial": 0,
        "no_data": 0,
        "error": 1,
        "not_attempted": 0,
    }
    assert [
        (row["row_number"], row["stage"], row["error_type"]) for row in findings.exceptions
    ] == [("3", "upstream", "ValidationError")]
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status_counts"]["exceptions"]["by_error_type"] == {"ValidationError": 1}
    assert manifest["api"]["data_http_attempts"] == manifest["api"]["auth_http_attempts"] == 0
    assert all(
        (output / name).is_file()
        for name in (
            "findings.json",
            "report.html",
            "flagged_subtier_entities.csv",
            "run_manifest.json",
        )
    )
    assert "ValidationError" in (output / "report.html").read_text(encoding="utf-8")
    assert "sensitive" not in caplog.text


@pytest.fixture
def audit_case(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, ResponseCache]:
    """Three independent suppliers plus a separate portfolio for audit status coverage."""
    monkeypatch.chdir(tmp_path)
    configured = settings.model_copy(
        update={
            "entity_file_path": tmp_path / "input.xlsx",
            "duckdb_path": tmp_path / "warehouse.duckdb",
            "hub_max_countries": 3,
            "max_upstream_depth": 2,
            "upstream_limit": 17,
        }
    )
    workbook = Workbook()
    workbook.remove(cast(Any, workbook.active))
    cache = ResponseCache(configured.cache_dir)
    rows: dict[str, list[tuple[str, str | None]]] = {
        "list_3": [("Supplier A", "a"), ("Supplier B", "b"), ("Supplier C", "c")],
        "audit_cases": [
            ("Additional A", "w"),
            ("Additional duplicate", "w"),
            ("Additional absent", None),
            ("Additional weak", "weak"),
            ("Additional invalid", "bad"),
        ],
    }
    for portfolio, inputs in rows.items():
        sheet = workbook.create_sheet(portfolio)
        sheet.append(["name", "address", "country"])
        for name, identifier in inputs:
            sheet.append([name, None, None])
            data = [
                {
                    "entity_id": identifier,
                    "label": identifier,
                    "match_strength": {"value": "weak" if identifier == "weak" else "strong"},
                }
            ]
            cache.put(
                cache.key(
                    httpx.Request(
                        "GET", configured.sayari_api_base + "/v1/resolution", params={"name": name}
                    )
                ),
                json.dumps(
                    resolution_payload([{}] if identifier == "bad" else data if identifier else []),
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            if identifier and identifier not in {"weak", "bad"}:
                cache.put(
                    cache.key(
                        httpx.Request(
                            "GET",
                            configured.sayari_api_base + f"/v1/entity_summary/{identifier}",
                            params={},
                        )
                    ),
                    json.dumps(
                        entity_payload(
                            {
                                "id": identifier,
                                "label": name,
                                "countries": [],
                                "risk": {},
                                "psa_count": 0,
                            }
                        ),
                        ensure_ascii=False,
                    ).encode("utf-8"),
                )
                cache.put(
                    cache.key(
                        httpx.Request(
                            "GET",
                            configured.sayari_api_base + f"/v1/supply_chain/upstream/{identifier}",
                            params={
                                "max_depth": configured.max_upstream_depth,
                                "limit": configured.upstream_limit,
                            },
                        ),
                    ),
                    json.dumps(
                        {
                            "filters": {},
                            "explored_count": 0,
                            "data": {"paths": [], "entities": {}},
                            "partial_results": False,
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                )
    workbook.save(configured.entity_file_path)
    workbook.close()
    return configured, cache


def _matches(stage: str, path: str, params: dict[str, Any] | None) -> bool:
    return (
        path == "/v1/resolution" and params == {"name": "Supplier B"}
        if stage == "resolution"
        else path == "/v1/entity_summary/b"
        if stage == "profile"
        else path == "/v1/supply_chain/upstream/b"
    )


@pytest.mark.parametrize("stage", ["resolution", "profile", "upstream"])
def test_one_row_failure_preserves_later_rows_and_earlier_success(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stage: str,
) -> None:
    # A failure in one row's stage neither erases earlier successes nor stops later rows.
    configured, _ = audit_case
    original = AuditedTransport.handle_request

    def fail_one(self: AuditedTransport, request: httpx.Request) -> httpx.Response:
        if _matches(stage, request.url.path, dict(request.url.params)):
            raise RuntimeError("private exception payload must never escape")
        return original(self, request)

    monkeypatch.setattr(AuditedTransport, "handle_request", fail_one)
    findings = pipeline.run_pipeline(configured, offline=True)
    first, failed, later = findings.suppliers
    assert first["profile_status"] == later["profile_status"] == "available"
    assert later["coverage_status"] == "no_data"
    assert len(findings.suppliers) == 3
    assert [(e["stage"], e["error_type"]) for e in findings.exceptions] == [(stage, "RuntimeError")]
    if stage == "resolution":
        assert failed["resolution_status"] == "error"
    else:
        assert failed["resolution_status"] == "resolved"
        assert failed["profile_status"] == ("error" if stage == "profile" else "available")
        assert failed["coverage_status"] == ("error" if stage == "upstream" else "no_data")
    manifest = json.loads(Path("data/processed/run_manifest.json").read_text())
    assert manifest["status_counts"]["exceptions"]["by_stage"] == {stage: 1}
    assert "private exception payload" not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("stage", ["resolution", "profile", "upstream"])
def test_offline_miss_writes_audit_before_terminal_raise(
    audit_case: tuple[Settings, ResponseCache], monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    # An offline cache miss writes the manifest and findings for the failure, then raises.
    configured, cache = audit_case
    path, params = {
        "resolution": ("/v1/resolution", {"name": "Supplier B"}),
        "profile": ("/v1/entity_summary/b", {}),
        "upstream": (
            "/v1/supply_chain/upstream/b",
            {
                "max_depth": configured.max_upstream_depth,
                "limit": configured.upstream_limit,
            },
        ),
    }[stage]
    (
        cache.cache_dir
        / (
            cache.key(
                httpx.Request(
                    "GET", configured.sayari_api_base + path, params=cast(dict[str, Any], params)
                )
            )
            + ".json"
        )
    ).unlink()
    send = Mock(side_effect=AssertionError("Offline network forbidden"))
    monkeypatch.setattr(AuditedTransport, "_send", send)
    with pytest.raises(OfflineCacheMiss):
        pipeline.run_pipeline(configured, offline=True)
    send.assert_not_called()
    manifest = json.loads(Path("data/processed/run_manifest.json").read_text())
    assert manifest["run"]["execution_mode"] == "offline"
    assert manifest["api"]["data_http_attempts"] == 0
    assert manifest["api"]["cache_lookups"] - manifest["api"]["cache_hits"] == 1
    assert manifest["status_counts"]["exceptions"]["by_error_type"] == {"OfflineCacheMiss": 1}
    findings = json.loads(Path("data/processed/findings.json").read_text())
    assert findings["suppliers"][-1]["profile_status"] == "available"
    assert findings["exceptions"][0]["stage"] == stage


@pytest.mark.parametrize("stage", ["resolution", "profile", "upstream"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_process_control_propagates(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error: type[BaseException],
) -> None:
    # KeyboardInterrupt and SystemExit propagate, and no run manifest is written.
    original = AuditedTransport.handle_request

    def interrupt(self: AuditedTransport, request: httpx.Request) -> httpx.Response:
        if _matches(stage, request.url.path, dict(request.url.params)):
            raise error()
        return original(self, request)

    monkeypatch.setattr(AuditedTransport, "handle_request", interrupt)
    with pytest.raises(error):
        pipeline.run_pipeline(audit_case[0], offline=True)
    assert not Path("data/processed/run_manifest.json").exists()


@pytest.mark.parametrize(
    "target, error",
    [
        ("build_warehouse", duckdb.CatalogException),
        ("_convergence", ValueError),
        ("classify_risk_factors", UnclassifiedRiskFactors),
        ("coverage_summary", ValueError),
        ("assemble_findings", ValueError),
        ("render_report", OSError),
    ],
)
def test_run_level_failure_propagates_without_success_audit(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    error: type[BaseException],
) -> None:
    # A run-wide failure propagates and leaves no run manifest claiming success.
    monkeypatch.setattr(pipeline, target, Mock(side_effect=error("synthetic invariant")))
    with pytest.raises(error):
        pipeline.run_pipeline(audit_case[0], offline=True)
    assert not Path("data/processed/run_manifest.json").exists()

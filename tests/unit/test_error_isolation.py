"""Task 13 proves existing row isolation without adding catch sites."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import duckdb
import httpx
import pytest

from sayari_poc import cli, pipeline
from sayari_poc.analysis import UnclassifiedRiskFactors
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import AuditedTransport, OfflineCacheMiss, RateLimitPacer, SayariAuthError
from tests.ontology_support import pipeline_ontology as pipeline_ontology
from tests.pipeline_support import _node, _upstream

pytestmark = pytest.mark.usefixtures("pipeline_ontology")


@pytest.mark.parametrize("entrypoint", ["pipeline", "cli"])
@pytest.mark.parametrize("failure", ["rejected", "overflow"])
def test_rejected_token_is_attempted_once_and_preserves_pipeline_outputs(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    entrypoint: str,
    failure: str,
) -> None:
    # Rejected or unusable tokens fail once, preserving artifacts and avoiding warehouse creation.
    configured, _ = audit_case
    output = tmp_path / "data/processed"
    output.mkdir(parents=True)
    sentinels = {
        name: f"previous {name}\r\n".encode()
        for name in ("findings.json", "report.html", "run_manifest.json")
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)
    paths: list[str] = []
    clients: list[SayariClient] = []

    def reject(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if failure == "overflow":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-token",
                    "expires_in": 10**100,
                    "token_type": "Bearer",
                },
            )
        return httpx.Response(401, json={"message": "synthetic-private-rejection"})

    def client_factory(settings: Settings, cache: ResponseCache, **kwargs: Any) -> SayariClient:
        client = SayariClient(
            settings,
            cache,
            transport=httpx.MockTransport(reject),
            pacer=RateLimitPacer(sleep=lambda _: None),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(pipeline, "SayariClient", client_factory)
    message = (
        "Sayari authentication failed; check SAYARI_CLIENT_ID and SAYARI_CLIENT_SECRET. "
        "No outputs were written."
    )
    if entrypoint == "pipeline":
        with pytest.raises(SayariAuthError) as caught:
            pipeline.run_pipeline(configured, refresh=True, output_dir=output)
        assert str(caught.value) == message
    else:
        monkeypatch.setattr(cli, "load_settings", lambda: configured)
        assert cli.main(["run", "--sheet", "list_3", "--refresh"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.endswith(f"Run failed: {message}\n")
        assert "synthetic-private-rejection" not in captured.err
    assert paths == ["/oauth/token"]
    assert clients[0].audit.auth_http_attempts == 1
    assert clients[0].audit.data_http_attempts == clients[0].audit.pacing_waits == 0
    assert not configured.duckdb_path.exists()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == sentinels
    assert "synthetic-private-rejection" not in caplog.text


@pytest.mark.parametrize("stage", ["resolution", "profile", "upstream"])
def test_authentication_row_failure_is_fatal_before_assembly(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    # Auth failure at any stage is fatal at the pipeline boundary while stages still isolate rows.
    configured, _ = audit_case
    output = tmp_path / "previous"
    output.mkdir()
    sentinels = {
        name: f"previous {name}".encode()
        for name in ("findings.json", "report.html", "run_manifest.json")
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)
    original = AuditedTransport.handle_request

    def fail_auth(self: AuditedTransport, request: httpx.Request) -> httpx.Response:
        if _matches(stage, request.url.path, dict(request.url.params)):
            raise SayariAuthError("synthetic-private-exception")
        return original(self, request)

    assembly = Mock(side_effect=AssertionError("Authentication must fail before assembly"))
    monkeypatch.setattr(AuditedTransport, "handle_request", fail_auth)
    monkeypatch.setattr(pipeline, "assemble_findings", assembly)
    with pytest.raises(SayariAuthError, match="No outputs were written"):
        pipeline.run_pipeline(configured, offline=True, output_dir=output)
    assembly.assert_not_called()
    assert not configured.duckdb_path.exists()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == sentinels


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
            "run_manifest.json",
        )
    )
    assert "ValidationError" in (output / "report.html").read_text(encoding="utf-8")
    assert "sensitive" not in caplog.text


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

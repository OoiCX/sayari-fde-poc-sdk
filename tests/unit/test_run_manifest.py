"""Task 13 audit projection, telemetry and safe logging contracts."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from sayari_poc import cli, pipeline
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.risk_taxonomy import RiskOntology
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import AuditedTransport, OfflineCacheMiss, RateLimitPacer
from tests.ontology_support import pipeline_ontology as pipeline_ontology
from tests.p3_support import entity_payload, resolution_payload
from tests.unit.test_error_isolation import audit_case as _audit_case

audit_case = _audit_case
pytestmark = pytest.mark.usefixtures("pipeline_ontology")


@pytest.mark.parametrize("target", ["load_settings", "plan_work", "run_pipeline"])
def test_cli_unexpected_value_error_never_discloses_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], target: str
) -> None:
    # The text of an unexpected ValueError never reaches the CLI's output.
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(
        cli, target, Mock(side_effect=ValueError("synthetic-private-value /private/input.xlsx"))
    )
    args = ["run", "--dry-run"] if target == "plan_work" else ["run", "--offline"]
    assert cli.main(args) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Run failed: invalid input or data (ValueError); details withheld.\n"


def test_cli_validation_locations_never_disclose_dynamic_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A configuration error redacts a dynamic validation key instead of echoing it.
    error = ValidationError.from_exception_data(
        "Synthetic",
        [{"type": "string_type", "loc": ("private-key", 2), "input": "private-input"}],
    )
    monkeypatch.setattr(cli, "load_settings", Mock(side_effect=error))
    assert cli.main(["run", "--offline"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Configuration error: check <redacted>.2\n"


STABLE_KEYS = [
    "sheets",
    "limit",
    "input_entities",
    "headline_portfolio",
    "hub_max_countries",
    "path_max_per_node",
    "max_upstream_depth",
    "upstream_limit",
    "taxonomy",
]


def _manifest() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("data/processed/run_manifest.json").read_text(encoding="utf-8")),
    )


def test_manifest_exact_projection_schema_order_and_logging(
    audit_case: tuple[Settings, ResponseCache],
    caplog: pytest.LogCaptureFixture,
    pipeline_ontology: RiskOntology,
) -> None:
    # The manifest holds exactly the declared execution fields, always in the same key order.
    configured, _ = audit_case
    caplog.set_level(logging.INFO, logger="sayari_poc")
    before = datetime.now(UTC)
    findings = pipeline.run_pipeline(configured, all_sheets=True, offline=True)
    manifest = _manifest()
    assert list(manifest) == [
        "run",
        "api",
        "bounds",
        "taxonomy",
        "status_counts",
    ]
    assert list(manifest["run"]) == [
        "timestamp",
        "execution_mode",
        "generated_at_source",
        "sheets",
        "limit",
        "input_entities",
    ]
    assert before <= datetime.fromisoformat(manifest["run"]["timestamp"]) <= datetime.now(UTC)
    assert manifest["run"]["execution_mode"] == "offline"
    assert manifest["run"]["generated_at_source"] == "observed"
    assert manifest["run"]["sheets"] == ["list_3", "audit_cases"]
    assert manifest["run"]["limit"] is None and manifest["run"]["input_entities"] == 8
    assert list(cast(dict[str, Any], findings.manifest)) == STABLE_KEYS
    assert cast(dict[str, Any], findings.manifest)["path_max_per_node"] == 6
    assert manifest["api"] == {
        "data_http_attempts": 0,
        "auth_http_attempts": 0,
        "retry_counts": {"auth": 0, "data": 0},
        "pacing_waits": 0,
        "pacing_wait_seconds": 0.0,
        "cache_hits": 16,
        "cache_lookups": 16,
        "cache_hit_rate": 1.0,
        "call_budget": configured.call_budget,
        "budget_exhausted": False,
    }
    assert manifest["bounds"] == {
        key: getattr(configured, key)
        for key in ["hub_max_countries", "max_upstream_depth", "upstream_limit"]
    }
    assert manifest["taxonomy"] == cast(dict[str, Any], findings.manifest)["taxonomy"]
    assert all(
        manifest["taxonomy"][key] == value for key, value in pipeline_ontology.parameters().items()
    )
    counts = manifest["status_counts"]
    assert counts["resolution"] == {"resolved": 5, "weak": 1, "no_match": 1, "error": 1}
    assert list(counts["resolution"]) == ["resolved", "weak", "no_match", "error"]
    assert counts["profile"] == {"available": 5, "error": 0, "not_requested": 3}
    assert counts["coverage"] == findings.coverage
    assert list(counts["coverage"]["list_3"]) == [
        "assessed",
        "partial",
        "no_data",
        "error",
        "not_attempted",
    ]
    assert counts["exceptions"] == {
        "by_stage": {"resolution": 3},
        "by_error_type": {"ValidationError": 1, "no_match": 1, "weak": 1},
    }
    raw = Path("data/processed/run_manifest.json").read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert (
        raw.decode() == json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    events = [
        record.getMessage().split(":")[0]
        for record in caplog.records
        if record.name == "sayari_poc.pipeline" and record.levelno == logging.INFO
    ]
    assert events == [
        "ingestion complete",
        "resolution complete",
        "enrichment complete",
        "upstream complete",
        "warehouse complete",
        "outputs complete",
    ]
    assert "max_pages" not in raw.decode() and str(configured.cache_dir) not in raw.decode()


@pytest.mark.parametrize("mode", ["live", "offline", "refresh"])
def test_counter_grains_retry_refresh_and_read_only(
    settings: Settings, cache: ResponseCache, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The counters keep logical cache lookups and hits separate from actual HTTP attempts.
    monkeypatch.setattr("sayari.core.http_client.time.sleep", lambda _: None)
    attempts = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "access_token": "synthetic-token",
                },
            )
        attempts.append(request)
        return httpx.Response(
            503 if len(attempts) == 1 else 200,
            json=entity_payload({"id": request.url.path.rsplit("/", 1)[1], "version": 1}),
        )

    cache.put(
        cache.key(
            httpx.Request("GET", settings.sayari_api_base + "/v1/entity_summary/cached", params={})
        ),
        json.dumps(entity_payload({"id": "cached", "version": 0}), ensure_ascii=False).encode(
            "utf-8"
        ),
    )
    client = SayariClient(
        settings,
        cache,
        offline=mode == "offline",
        refresh=mode == "refresh",
        transport=httpx.MockTransport(respond),
        pacer=RateLimitPacer(sleep=lambda _: None, monotonic=lambda: 0.0),
    )
    client.get_entity("cached")
    if mode == "offline":
        with pytest.raises(OfflineCacheMiss):
            client.get_entity("missing")
        expected = (0, 1, 2)
    else:
        client.get_entity("missing")
        expected = (3, 0, 0) if mode == "refresh" else (2, 1, 2)
    assert (client.calls_made, client.cache_hits, client.cache_lookups) == expected
    for property_name in ["calls_made", "cache_hits", "cache_lookups"]:
        with pytest.raises(AttributeError):
            setattr(client, property_name, 99)


def test_atomic_manifest_write_failure_preserves_previous_and_fails_cli(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # If replacing the manifest fails, the old manifest stays in place and the CLI exits with 1.
    configured, _ = audit_case
    pipeline.run_pipeline(configured, offline=True)
    old = Path("data/processed/run_manifest.json").read_bytes()
    original = Path.replace

    def fail_replace(self: Path, target: Path) -> Path:
        if Path(target).name == "run_manifest.json":
            raise OSError("private path must not be printed")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        pipeline.run_pipeline(configured, offline=True)
    monkeypatch.setattr(cli, "load_settings", lambda: configured)
    assert cli.main(["run", "--offline"]) == 1
    captured = capsys.readouterr()
    assert "Findings:" not in captured.out and "private path" not in captured.err
    assert Path("data/processed/run_manifest.json").read_bytes() == old
    assert not Path("data/processed/run_manifest.tmp").exists()


@pytest.mark.parametrize("refresh", [False, True])
def test_real_client_budget_continues_cached_work_and_cli_names_shortfall(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    refresh: bool,
) -> None:
    # Once the call budget runs out, cached work still completes and the CLI reports the shortfall.
    configured, cache = audit_case
    configured.call_budget = 1
    (
        cache.cache_dir
        / (
            cache.key(
                httpx.Request(
                    "GET",
                    configured.sayari_api_base + "/v1/resolution",
                    params={"name": "Supplier A"},
                )
            )
            + ".json"
        )
    ).unlink()
    (
        cache.cache_dir
        / (
            cache.key(
                httpx.Request(
                    "GET",
                    configured.sayari_api_base + "/v1/resolution",
                    params={"name": "Supplier B"},
                )
            )
            + ".json"
        )
    ).unlink()
    real_client = SayariClient
    clients = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "access_token": "synthetic-budget-token",
                },
            )
        return httpx.Response(
            200,
            json=resolution_payload(
                [
                    {
                        "entity_id": "a",
                        "label": "a",
                        "match_strength": {"value": "strong"},
                    }
                ]
            ),
        )

    def make_client(*args: Any, **kwargs: Any) -> SayariClient:
        client = real_client(*args, **kwargs, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(pipeline, "SayariClient", make_client)
    monkeypatch.setattr(cli, "load_settings", lambda: configured)
    assert cli.main(["run", *(["--refresh"] if refresh else [])]) == 1
    assert "Budget exhausted:" in capsys.readouterr().err
    manifest = _manifest()
    api = manifest["api"]
    assert api["budget_exhausted"] and api["data_http_attempts"] == 1
    assert api["call_budget"] == 1
    assert manifest["run"]["execution_mode"] == ("refresh" if refresh else "live")
    assert api["cache_hit_rate"] == (None if refresh else round(5 / 7, 4))
    assert api["cache_lookups"] == (0 if refresh else 7)
    expected_denied = 4 if refresh else 1
    assert manifest["status_counts"]["exceptions"]["by_error_type"] == {
        "BudgetExceeded": expected_denied,
    }
    rows = json.loads(Path("data/processed/findings.json").read_text())["suppliers"]
    if not refresh:
        assert rows[1]["error_type"] == "BudgetExceeded"
        assert rows[2]["resolution_status"] == "resolved"
        assert rows[2]["profile_status"] == "available"
        assert rows[2]["coverage_status"] == "no_data"
    assert clients[0].calls_made == 1


def test_replay_only_changes_execution_fields(
    audit_case: tuple[Settings, ResponseCache], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A replay changes only the execution telemetry; the output files stay byte-identical.
    configured, _ = audit_case
    moments = iter(
        [
            datetime(2026, 9, 17, 1, tzinfo=UTC),
            datetime(2026, 9, 17, 2, tzinfo=UTC),
            datetime(2026, 9, 17, 3, tzinfo=UTC),
            datetime(2026, 9, 17, 4, tzinfo=UTC),
        ]
    )
    clock = Mock()
    clock.now.side_effect = lambda _: next(moments)
    monkeypatch.setattr(pipeline, "datetime", clock)
    pipeline.run_pipeline(configured, all_sheets=True, offline=False)
    paths = [
        Path("data/processed") / name
        for name in ["findings.json", "report.html", "flagged_subtier_entities.csv"]
    ]
    before = [path.read_bytes() for path in paths]
    first = _manifest()
    pipeline.run_pipeline(configured, all_sheets=True, offline=True)
    second = _manifest()
    assert before == [path.read_bytes() for path in paths]
    assert first["run"].pop("timestamp") != second["run"].pop("timestamp")
    assert first["run"].pop("execution_mode") == "live"
    assert second["run"].pop("execution_mode") == "offline"
    first.pop("api")
    second.pop("api")
    assert first == second


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), Path("private"), SecretStr("fake")]
)
def test_manifest_rejects_non_json_before_replacing_existing(
    tmp_path: Path, invalid: object
) -> None:
    # Metadata that cannot be serialized to JSON never overwrites an existing manifest.
    path = tmp_path / "run_manifest.json"
    path.write_text("previous audit", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        pipeline._write_run_manifest({"invalid": invalid}, path)
    assert path.read_text() == "previous audit"
    assert not path.with_suffix(".tmp").exists()


def test_cli_logging_is_scoped_and_dry_run_streams_remain_clean(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Each CLI call restores the logging setup it found, and dry-run output stays clean JSON.
    configured, _ = audit_case
    monkeypatch.setattr(cli, "load_settings", lambda: configured)
    loggers = [logging.getLogger(name) for name in ["", "sayari_poc", "httpx", "httpcore"]]
    before = [(list(logger.handlers), logger.level, logger.propagate) for logger in loggers]
    for _ in range(2):
        assert cli.main(["run", "--offline"]) == 0
        output = capsys.readouterr()
        assert output.err.count("ingestion complete:") == 1
        assert "ingestion complete" not in output.out
    assert cli.main(["run", "--dry-run"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out.splitlines()[0])["input_entities"] == 3
    assert not output.err
    assert any(r.name == "sayari_poc.pipeline" for r in caplog.records)
    assert [(list(logger.handlers), logger.level, logger.propagate) for logger in loggers] == before


def test_synthetic_credentials_and_unsafe_messages_never_reach_outputs(
    audit_case: tuple[Settings, ResponseCache],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Synthetic credentials and unsafe error text never appear in console output, logs or files.
    configured, cache = audit_case
    secrets = ['synthetic-id-"\\\n-end', 'synthetic-secret-"\\\n-end', 'synthetic-bearer-"\\\n-end']
    configured.sayari_client_id = secrets[0]
    configured.sayari_client_secret = SecretStr(secrets[1])
    original = SayariClient
    original_get = AuditedTransport.handle_request

    def raise_private(self: AuditedTransport, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/entity_summary/b":
            raise RuntimeError("raw-private-response " + " ".join(secrets))
        return original_get(self, request)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"expires_in": 3600, "token_type": "Bearer", "access_token": secrets[2]}
            )
        # The real boundary must reject the echoed bearer, including escaped characters.
        return httpx.Response(200, json={**resolution_payload([]), "echo": secrets[2]})

    (
        cache.cache_dir
        / (
            cache.key(
                httpx.Request(
                    "GET",
                    configured.sayari_api_base + "/v1/resolution",
                    params={"name": "Supplier A"},
                )
            )
            + ".json"
        )
    ).unlink()
    monkeypatch.setattr(AuditedTransport, "handle_request", raise_private)
    monkeypatch.setattr(
        pipeline,
        "SayariClient",
        lambda *a, **k: original(*a, **k, transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(cli, "load_settings", lambda: configured)
    assert cli.main(["run"]) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err + caplog.text
    for path in Path("data/processed").iterdir():
        output += path.read_text(encoding="utf-8-sig")
    unsafe = secrets + [json.dumps(secret)[1:-1] for secret in secrets] + ["raw-private-response"]
    assert not any(secret in output for secret in unsafe)
    assert not any(record.exc_info for record in caplog.records)
    assert _manifest()["status_counts"]["exceptions"]["by_error_type"] == {
        "RuntimeError": 1,
        "SayariError": 1,
    }


@pytest.mark.parametrize("filename", ["run_manifest.json", "run_manifest.tmp"])
def test_manifest_cannot_overwrite_workbook(
    audit_case: tuple[Settings, ResponseCache], filename: str
) -> None:
    # Writing the manifest never overwrites the input workbook.
    configured, _ = audit_case
    output_dir = configured.entity_file_path.parent
    configured.entity_file_path = output_dir / filename
    configured.entity_file_path.write_bytes(b"protected input")
    with pytest.raises(ValueError, match="overwrite"):
        pipeline.run_pipeline(configured, output_dir=output_dir, offline=True)
    assert configured.entity_file_path.read_bytes() == b"protected input"


@pytest.mark.parametrize(
    ("sheet", "message"),
    [
        ("synthetic-private-sheet", "Selected sheet not found in workbook"),
        ("bad_headers", "Selected sheet: missing headers: country"),
        ("", "Select at least one nonempty sheet name"),
    ],
)
def test_cli_known_input_errors_are_useful_without_echoing_sheet_names(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sheet: str,
    message: str,
) -> None:
    # Known input errors still give useful guidance without echoing the sheet name.
    source = Path(__file__).resolve().parents[1] / "fixtures/mini_list.xlsx"
    configured = Settings(_env_file=None, entity_file_path=source)
    monkeypatch.setattr(cli, "load_settings", lambda: configured)
    assert cli.main(["run", "--dry-run", "--sheet", sheet]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"Run failed: {message}\n"

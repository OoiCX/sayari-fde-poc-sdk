"""Sequence the existing ingestion, resolution, enrichment, and output stages."""

import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
from pydantic import ValidationError

from sayari_poc.analysis import (
    build_warehouse,
    classify_risk_factors,
    convergence_funnel,
    coverage_summary,
    sensitivity_sweep,
    shared_nodes,
    suppressed_hubs,
)
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.enrich import fetch_profiles
from sayari_poc.excel import InputError, read_all_sheets, read_sheet
from sayari_poc.findings import assemble_findings
from sayari_poc.identity import psa_exposure
from sayari_poc.models import (
    EntityProfile,
    Findings,
    IngestResult,
    ResolvedEntity,
    UpstreamResult,
)
from sayari_poc.report import render_report
from sayari_poc.resolve import resolve_entities
from sayari_poc.risk_taxonomy import RiskOntology, load_ontology
from sayari_poc.rollups import supplier_breakdown
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import OfflineCacheMiss
from sayari_poc.upstream import fetch_upstreams

OUTPUT_DIR = Path("data/processed")
DEFAULT_SHEET = "list_3"
SUPPRESSED_HUB_ROWS = 10
# Measure how sensitive the result is to the country limit at these thresholds. The sweep is kept in
# findings.json for audit, but the report does not render it.
SENSITIVITY_THRESHOLDS = (5, 10, 15, 20)
_LOGGER = logging.getLogger(__name__)


def _ingest(
    settings: Settings,
    sheets: list[str] | None,
    limit: int | None,
    all_sheets: bool,
) -> dict[str, IngestResult]:
    """Apply one retained-entity cap across selected portfolios."""
    if limit is not None and limit <= 0:
        raise InputError("limit must be a positive integer")
    if all_sheets and sheets is not None:
        raise InputError("Choose sheets or all_sheets, not both")
    if sheets == [] or (sheets is not None and any(not sheet.strip() for sheet in sheets)):
        raise InputError("Select at least one nonempty sheet name")
    if all_sheets:
        return read_all_sheets(settings.entity_file_path, limit=limit)
    results: dict[str, IngestResult] = {}
    remaining = limit
    for sheet in dict.fromkeys(sheets if sheets is not None else [DEFAULT_SHEET]):
        result = read_sheet(settings.entity_file_path, sheet, limit=remaining)
        results[sheet] = result
        if remaining is not None:
            # Only retained entities use up the allowance; ingestion diagnostics don't.
            remaining -= len(result.entities)
    return results


def plan_work(
    settings: Settings,
    sheets: list[str] | None = None,
    limit: int | None = None,
    *,
    all_sheets: bool = False,
) -> dict[str, object]:
    """Estimate logical requests from local inputs without retrieval."""
    ingested = _ingest(settings, sheets, limit, all_sheets)
    resolutions = sum(len(result.entities) for result in ingested.values())
    profiles = resolutions
    return {
        "sheets": list(ingested),
        "input_entities": resolutions,
        "resolution_requests": resolutions,
        "profile_requests_up_to": profiles,
        "upstream_requests_up_to": profiles,
        "planned_calls": resolutions + 2 * profiles,
        "ingestion_diagnostics": sum(len(result.exceptions) for result in ingested.values()),
    }


def _write_findings(findings: Findings, path: Path, *, reuse_generated_at: bool = True) -> None:
    """Write stable JSON with optional reuse of unchanged content time.

    When reuse is allowed, update findings.generated_at in place from a valid previous artifact;
    invalid or substantively different artifacts are ignored.
    """
    # generated_at marks when the artifact was generated, not every replay of unchanged evidence.
    # Reuse the old value only when everything else matches the previous valid artifact.
    if reuse_generated_at:
        try:
            previous = Findings.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            previous = None
        if previous is not None and previous.model_dump(
            exclude={"generated_at"}
        ) == findings.model_dump(
            exclude={"generated_at"},
        ):
            findings.generated_at = previous.generated_at
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(findings.model_dump(mode="json"), ensure_ascii=False, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


# Keep execution telemetry out of Findings, so the evidence stays byte-stable.
def _run_manifest(
    findings: Findings,
    client: SayariClient,
    settings: Settings,
    *,
    offline: bool,
    refresh: bool,
) -> dict[str, object]:
    """Project execution telemetry without changing Findings."""
    resolution = Counter(row["resolution_status"] for row in findings.suppliers)
    profile = Counter(row["profile_status"] for row in findings.suppliers)
    by_stage = Counter(row["stage"] for row in findings.exceptions if "stage" in row)
    by_error_type = Counter(row["error_type"] for row in findings.exceptions if "error_type" in row)
    # No cache lookups means an unknown hit ratio, not a measured zero.
    return {
        "run": {
            "timestamp": datetime.now(UTC).isoformat(),
            "execution_mode": "offline" if offline else "refresh" if refresh else "live",
            "generated_at_source": "declared"
            if settings.declared_generated_at is not None
            else "observed",
            **{key: findings.manifest[key] for key in ("sheets", "limit", "input_entities")},
        },
        "api": {
            "data_http_attempts": client.calls_made,
            "auth_http_attempts": client.audit.auth_http_attempts,
            "retry_counts": {
                kind: sum(
                    attempt.retry for attempt in client.audit.attempts if attempt.kind == kind
                )
                for kind in ("auth", "data")
            },
            "pacing_waits": client.audit.pacing_waits,
            "pacing_wait_seconds": client.audit.pacing_wait_seconds,
            "cache_hits": client.cache_hits,
            "cache_lookups": client.cache_lookups,
            "cache_hit_rate": round(client.cache_hits / client.cache_lookups, 4)
            if client.cache_lookups
            else None,
            "call_budget": settings.call_budget,
            "budget_exhausted": by_error_type["BudgetExceeded"] > 0,
        },
        "bounds": {
            key: findings.manifest[key]
            for key in ("hub_max_countries", "max_upstream_depth", "upstream_limit")
        },
        "taxonomy": findings.manifest["taxonomy"],
        "status_counts": {
            "resolution": {
                key: resolution[key] for key in ("resolved", "weak", "no_match", "error")
            },
            "profile": {key: profile[key] for key in ("available", "error", "not_requested")},
            "coverage": findings.coverage,
            "exceptions": {
                "by_stage": dict(sorted(by_stage.items())),
                "by_error_type": dict(sorted(by_error_type.items())),
            },
        },
    }


def _write_run_manifest(manifest: dict[str, object], path: Path) -> None:
    """Publish execution metadata atomically."""
    serialized = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _convergence(
    con: duckdb.DuckDBPyConnection, portfolios: list[str], threshold: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Reconcile each portfolio's ranked nodes and selection counts."""
    ranked: list[dict[str, Any]] = []
    convergence: dict[str, dict[str, Any]] = {}
    for portfolio in sorted(portfolios):
        # Ranking and evidence come from one query, so we can't get orphan ranks or supplier
        # memberships built from different populations.
        nodes = shared_nodes(con, threshold, portfolio).to_dict("records")
        for node in nodes:
            if len(node["suppliers"]) != node["supplier_count"]:
                raise ValueError("Connected supplier evidence does not reconcile")
        funnel = convergence_funnel(con, threshold, portfolio)
        if funnel["after_severity_filter"] != len(nodes):
            raise ValueError("Convergence funnel differs from ranked survivors")
        ranked.extend(nodes)
        # Always add the configured threshold to the fixed ones, so the sweep's value there equals
        # the funnel's final stage by construction rather than by coincidence.
        sensitivity = sensitivity_sweep(con, portfolio, {*SENSITIVITY_THRESHOLDS, threshold})
        convergence[portfolio] = {
            "funnel": funnel,
            "sensitivity": sensitivity,
            "suppressed_hubs": suppressed_hubs(
                con,
                threshold,
                portfolio,
                SUPPRESSED_HUB_ROWS,
            ).to_dict("records"),
        }
    return ranked, convergence


def _attach_rollups(
    convergence: dict[str, dict[str, Any]],
    resolved: list[ResolvedEntity],
    profiles: dict[str, EntityProfile],
    upstream: dict[str, UpstreamResult],
    ranked: list[dict[str, Any]],
    ontology: RiskOntology,
) -> None:
    """Attach supplier worklists to existing convergence records."""
    for portfolio, summary in convergence.items():
        nodes = [node for node in ranked if node["portfolio"] == portfolio]
        summary["suppliers_ranked"] = supplier_breakdown(
            portfolio, resolved, profiles, upstream, nodes, ontology
        )


def run_pipeline(
    settings: Settings,
    sheets: list[str] | None = None,
    limit: int | None = None,
    *,
    all_sheets: bool = False,
    offline: bool = False,
    refresh: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> Findings:
    """Run sequential screening and publish all four artifacts.

    The global limit counts named entities in selected-sheet order, then Excel row order. All
    selected headers validate, even beyond the limit. Duplicate identities share canonical
    retrieval while each input row remains accounted for. The output directory and warehouse are
    created as needed.

    Args:
        settings: Input paths, retrieval bounds, and optional declared output time.
        limit: Maximum retained entities across all selected sheets, or None.
        refresh: Bypass cached responses; incompatible with offline.
        offline: Require cached evidence and prohibit network requests.
        sheets: Ordered worksheet selection; defaults to list_3.
        all_sheets: Read every worksheet in workbook order instead of sheets.
        output_dir: Destination for Findings, HTML, and execution manifest.

    Returns:
        Completed evidence, including isolated row failures.

    Raises:
        ValueError: Options conflict or an output would overwrite the workbook.
        OfflineCacheMiss: Required cached evidence is absent. Error artifacts are written before
            this failure is raised.
    """
    if offline and refresh:
        raise InputError("offline and refresh are incompatible")
    for name in (
        "findings.json",
        "report.html",
        "run_manifest.json",
        "run_manifest.tmp",
    ):
        # Compare resolved paths so no output file can point at the source workbook.
        if (output_dir / name).resolve() == settings.entity_file_path.resolve():
            raise InputError("Output must not overwrite the input workbook")
    if settings.duckdb_path.resolve() == settings.entity_file_path.resolve():
        raise InputError("Warehouse must not overwrite the input workbook")
    ingested = _ingest(settings, sheets, limit, all_sheets)
    _LOGGER.info(
        "ingestion complete: sheets=%d rows=%d diagnostics=%d",
        len(ingested),
        sum(len(result.entities) for result in ingested.values()),
        sum(len(result.exceptions) for result in ingested.values()),
    )
    # Fail on a missing or invalid ontology before we build any retrieval client.
    ontology = load_ontology()
    inputs = [entity for result in ingested.values() for entity in result.entities]
    client = SayariClient(
        settings,
        ResponseCache(settings.cache_dir),
        offline=offline,
        refresh=refresh,
    )
    try:
        resolved = resolve_entities(client, inputs)
        resolution_counts = Counter(row.status for row in resolved)
        _LOGGER.info(
            "resolution complete: resolved=%d weak=%d no_match=%d error=%d",
            *(resolution_counts[key] for key in ("resolved", "weak", "no_match", "error")),
        )
        profiles = fetch_profiles(client, resolved)
        _LOGGER.info(
            "enrichment complete: profiles_available=%d profile_errors=%d",
            len(profiles),
            sum(bool(row.profile_error) for row in resolved),
        )
        upstream = fetch_upstreams(client, resolved, settings)
        upstream_counts = Counter(result.status for result in upstream.values())
        _LOGGER.info(
            "upstream complete: suppliers_attempted=%d assessed=%d partial=%d no_data=%d error=%d",
            len(upstream),
            *(upstream_counts[key] for key in ("assessed", "partial", "no_data", "error")),
        )
    finally:
        client.close()
    portfolios = list(ingested)
    headline = DEFAULT_SHEET if DEFAULT_SHEET in portfolios else next(iter(portfolios), None)
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = build_warehouse(settings.duckdb_path, resolved, profiles, upstream)
    try:
        classify_risk_factors(con, ontology)
        ranked, convergence = _convergence(con, portfolios, settings.hub_max_countries)
        _attach_rollups(convergence, resolved, profiles, upstream, ranked, ontology)
        coverage = coverage_summary(con)
    finally:
        con.close()
    _LOGGER.info(
        "warehouse complete: supplier_rows=%d shared_nodes=%d",
        len(resolved),
        len(ranked),
    )
    psa_records = psa_exposure(resolved, profiles, ontology).to_dict("records")
    findings = assemble_findings(
        generated_at=settings.declared_generated_at
        if settings.declared_generated_at is not None
        else datetime.now(UTC).isoformat(),
        ingested=ingested,
        inputs=inputs,
        resolved=resolved,
        profiles=profiles,
        upstream=upstream,
        psa_records=psa_records,
        ranked=ranked,
        convergence=convergence,
        coverage=coverage,
        headline_portfolio=headline,
        limit=limit,
        settings=settings,
        ontology=ontology,
    )
    _write_findings(
        findings,
        output_dir / "findings.json",
        reuse_generated_at=settings.declared_generated_at is None,
    )
    render_report(findings, output_dir / "report.html")
    _write_run_manifest(
        _run_manifest(
            findings,
            client,
            settings,
            offline=offline,
            refresh=refresh,
        ),
        output_dir / "run_manifest.json",
    )
    _LOGGER.info(
        "outputs complete: artifacts_written=3 exception_rows=%d", len(findings.exceptions)
    )
    # Stages isolate row failures, but the run as a whole must still fail loudly on cache misses.
    if any(item.get("error_type") == "OfflineCacheMiss" for item in findings.exceptions):
        raise OfflineCacheMiss(
            "Offline cache miss: required responses are absent; see generated exceptions. "
            "No network fallback was attempted."
        )
    return findings

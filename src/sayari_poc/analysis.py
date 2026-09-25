"""Analyze retrieved evidence in DuckDB without API access."""

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd  # type: ignore[import-untyped]

from sayari_poc.models import (
    EntityProfile,
    ResolvedEntity,
    RiskFactor,
    UpstreamEntity,
    UpstreamResult,
)
from sayari_poc.risk_taxonomy import SELECTED_LEVELS, RiskOntology, load_ontology


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Recreate tables within the caller's transaction."""
    con.execute(Path(__file__).with_name("sql").joinpath("schema.sql").read_text(encoding="utf-8"))


def _load_suppliers(
    con: duckdb.DuckDBPyConnection,
    suppliers: list[ResolvedEntity],
    profiles: dict[str, EntityProfile],
    upstream: dict[str, UpstreamResult],
) -> set[str]:
    """Load unique workbook rows and return accepted supplier IDs."""
    rows: dict[tuple[str, int], ResolvedEntity] = {}
    resolved_ids: set[str] = set()
    for row in suppliers:
        # Key supplier rows by workbook position, not canonical ID.
        key = (row.sheet, row.row_number)
        if key in rows and rows[key] != row:
            raise ValueError(f"Conflicting supplier input row: {key!r}")
        rows[key] = row
        if row.status == "resolved":
            if row.entity_id is None or not row.entity_id.strip():
                raise ValueError(f"Resolved supplier lacks a canonical ID: {key!r}")
            resolved_ids.add(row.entity_id)

    for key in sorted(rows):
        row = rows[key]
        # A weak candidate must not pick up the assessment of a resolved row with the same ID.
        profile = profiles.get(row.entity_id or "") if row.status == "resolved" else None
        retrieval = upstream.get(row.entity_id or "") if row.status == "resolved" else None
        con.execute(
            "INSERT INTO suppliers (portfolio, row_number, entity_id, input_name, label, "
            "translated_label, match_strength, resolution_status, coverage_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                row.sheet,
                row.row_number,
                row.entity_id,
                row.input_name,
                profile.label if profile is not None and profile.label is not None else row.label,
                profile.translated_label
                if profile is not None and profile.translated_label is not None
                else row.translated_label,
                row.match_strength,
                row.status,
                retrieval.status if retrieval is not None else None,
            ],
        )
    return resolved_ids


def _load_upstream(
    con: duckdb.DuckDBPyConnection,
    upstream: dict[str, UpstreamResult],
    resolved_ids: set[str],
) -> None:
    """Load canonical entities and distinct non-self supplier edges."""
    entities: dict[str, UpstreamEntity] = {}
    edges: set[tuple[str, str]] = set()
    for supplier_id, retrieval in sorted(upstream.items()):
        if supplier_id != retrieval.supplier_id or supplier_id not in resolved_ids:
            raise ValueError(f"Upstream result must match a resolved supplier ID: {supplier_id!r}")
        if retrieval.status in {"no_data", "error"} and retrieval.entities:
            raise ValueError(
                f"Upstream {retrieval.status} result contains entities: {supplier_id!r}"
            )
        # Entities are validated inside each supplier's retrieval. Doing it again here would turn
        # one bad response into a failure of the whole run.
        for entity_id, entity in sorted(retrieval.entities.items()):
            # fetch_upstream returns the response as received. Sort here instead, because this copy
            # is what reaches the warehouse and its order must be stable.
            entity = entity.model_copy(
                update={
                    "countries": sorted(entity.countries),
                    "risk_factors": sorted(entity.risk_factors),
                }
            )
            if entity_id in entities and entities[entity_id] != entity:
                # Defence in depth: fetch_upstreams already shares observations and turns conflicts
                # into supplier errors, so the pipeline should never reach this guard.
                raise ValueError(f"Conflicting upstream evidence for canonical ID: {entity_id!r}")
            entities[entity_id] = entity
            # Keep the traversal root as an entity, but don't invent an edge from it to itself.
            if entity_id != supplier_id:
                edges.add((supplier_id, entity_id))

    for entity_id, entity in sorted(entities.items()):
        con.execute(
            "INSERT INTO upstream_entities "
            "(upstream_id, label, translated_label, countries, country_count) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                entity_id,
                entity.label,
                entity.translated_label,
                entity.countries,
                entity.country_count,
            ],
        )
        for factor in sorted(set(entity.risk_factors)):
            con.execute(
                "INSERT INTO risk_factors (entity_id, factor, source, level) "
                "VALUES (?, ?, 'upstream', NULL)",
                [entity_id, factor],
            )
    if edges:
        con.executemany(
            "INSERT INTO supplier_upstream (supplier_id, upstream_id) VALUES (?, ?)", sorted(edges)
        )


def _load_supply_paths(
    con: duckdb.DuckDBPyConnection,
    upstream: dict[str, UpstreamResult],
) -> None:
    """Retain source path order and Sayari tier annotations."""
    rows: list[tuple[object, ...]] = []
    for supplier_id, retrieval in sorted(upstream.items()):
        if retrieval.status in {"no_data", "error"} and retrieval.paths:
            raise ValueError(f"Upstream {retrieval.status} result contains paths: {supplier_id!r}")
        for path in retrieval.paths:
            if path.source_entity_id != supplier_id:
                raise ValueError(f"Upstream path source differs from supplier ID: {supplier_id!r}")
            # Store each hop's position separately from the tier annotation Sayari gave it.
            for hop_position, hop in enumerate(path.hops):
                rows.append(
                    (
                        path.source_entity_id,
                        path.path_index,
                        hop_position,
                        hop.tier,
                        hop.entity_id,
                        path.hops[-1].entity_id,
                        [component.model_dump() for component in hop.components],
                    )
                )
    if rows:
        con.executemany(
            "INSERT INTO supply_paths "
            "(source_entity_id, path_index, hop_position, tier, entity_id, "
            "terminal_entity_id, components) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _load_profile_risks(
    con: duckdb.DuckDBPyConnection,
    profiles: dict[str, EntityProfile],
    resolved_ids: set[str],
) -> None:
    """Enrich factor values without losing upstream provenance."""
    for entity_id, profile in sorted(profiles.items()):
        if entity_id != profile.entity_id or entity_id not in resolved_ids:
            raise ValueError(f"Profile must match a resolved supplier ID: {entity_id!r}")
        factors: dict[str, RiskFactor] = {}
        for factor in profile.risk_factors:
            if factor.factor in factors and factors[factor.factor] != factor:
                raise ValueError(
                    f"Conflicting profile factor evidence: {(entity_id, factor.factor)!r}"
                )
            factors[factor.factor] = factor
        for name, factor in sorted(factors.items()):
            # When a factor is in both, keep the upstream source so traversal-reported factors still
            # qualify, and take only the raw level from the profile. Factors seen only in the
            # profile keep 'profile' as their source.
            con.execute(
                "INSERT INTO risk_factors (entity_id, factor, source, level) "
                "VALUES (?, ?, 'profile', ?) ON CONFLICT (entity_id, factor) "
                "DO UPDATE SET level=excluded.level",
                [entity_id, name, factor.level],
            )


def build_warehouse(
    db_path: Path,
    suppliers: list[ResolvedEntity],
    profiles: dict[str, EntityProfile],
    upstream: dict[str, UpstreamResult],
) -> duckdb.DuckDBPyConnection:
    """Rebuild atomically and return a caller-owned open connection.

    Identical repeated input rows, nodes, edges and factors collapse at their declared grain.
    Conflicting same-key evidence raises ValueError, except upstream country/factor list
    ordering, which is canonicalized on copies. On overlap, upstream provenance remains and the
    profile supplies the raw level. Classification remains NULL. On failure the previous
    warehouse survives and the connection closes.
    """
    con = duckdb.connect(str(db_path))
    try:
        con.begin()
        create_schema(con)
        resolved_ids = _load_suppliers(con, suppliers, profiles, upstream)
        _load_upstream(con, upstream, resolved_ids)
        _load_supply_paths(con, upstream)
        _load_profile_risks(con, profiles, resolved_ids)
        con.commit()
    except BaseException:
        # Closing an uncommitted DuckDB connection rolls back the DDL as well as inserted rows.
        con.close()
        raise
    return con


class UnclassifiedRiskFactors(RuntimeError):
    """An incomplete classification must not masquerade as an empty finding."""


def classify_risk_factors(
    con: duckdb.DuckDBPyConnection, ontology: RiskOntology | None = None
) -> int:
    """Classify all loaded factors without altering raw evidence.

    Reapply even previously classified values so taxonomy changes need no reload. Identical
    slugs share an outcome; updating by slug avoids per-entity repetition. One UPDATE makes
    reclassification atomic even in autocommit mode. Transaction ownership stays with the
    caller, as for create_schema.
    """
    ontology = load_ontology() if ontology is None else ontology
    factors = con.execute(
        "SELECT factor, COUNT(*) FROM risk_factors GROUP BY factor ORDER BY factor"
    ).fetchall()
    if not factors:
        return 0
    classifications = [ontology.factors.get(factor) for factor, _ in factors]
    # The parallel arrays line up on the sorted slug list: classify each slug once, then join back
    # to every entity that has it, without multiplying rows by the risk-factor count.
    con.execute(
        """
        UPDATE risk_factors AS r
        SET is_severe = c.is_severe, ontology_level = c.ontology_level,
            risk_type = c.risk_type, ontology_status = c.ontology_status
        FROM (
            SELECT unnest($factors::TEXT[]) AS factor,
                   unnest($severe::BOOLEAN[]) AS is_severe,
                   unnest($levels::TEXT[]) AS ontology_level,
                   unnest($types::TEXT[]) AS risk_type,
                   unnest($statuses::TEXT[]) AS ontology_status
        ) AS c
        WHERE r.factor = c.factor
        """,
        {
            "factors": [factor for factor, _ in factors],
            "severe": [item.level in SELECTED_LEVELS if item else None for item in classifications],
            "levels": [item.level if item else None for item in classifications],
            "types": [item.risk_type if item else None for item in classifications],
            "statuses": ["resolved" if item else "unresolved" for item in classifications],
        },
    )
    return sum(count for _, count in factors)


def shared_nodes(
    con: duckdb.DuckDBPyConnection, hub_max_countries: int, portfolio: str
) -> pd.DataFrame:
    """Return ranked convergence within one portfolio's evidence.

    Counts are lower bounds; partial retrieval still contributes. Each row contains supplier
    identities and traversal-reported factors so consumers need no second evidence join.
    Coverage is reported separately.
    """
    # A node exactly at the country limit passes the SQL filter; only nodes above it are excluded.
    return _support_query(
        con,
        "node_evidence.sql",
        {"portfolio": portfolio, "hub_max_countries": hub_max_countries},
    )


def coverage_summary(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, int]]:
    """Count supplier input rows independently per portfolio.

    Assessed means within configured depth/result bounds. Partial, successful-empty (no_data),
    and upstream failure (error) remain distinct. NULL becomes the summary key not_attempted;
    existing resolution/profile diagnostics explain why. Duplicate resolved IDs still represent
    separate input rows. Absent portfolios stay absent.
    """
    outcomes = ("assessed", "partial", "no_data", "error")
    summary: dict[str, dict[str, int]] = {}
    rows = con.execute(
        "SELECT portfolio, coverage_status, COUNT(*) FROM suppliers "
        "GROUP BY portfolio, coverage_status ORDER BY portfolio, coverage_status"
    ).fetchall()
    for portfolio, status, count in rows:
        if status is not None and status not in outcomes:
            raise ValueError(f"Unrecognised coverage_status: {status!r}")
        if portfolio not in summary:
            summary[portfolio] = dict.fromkeys((*outcomes, "not_attempted"), 0)
        # No coverage status means retrieval wasn't attempted; it never means a successful no_data.
        key = "not_attempted" if status is None else status
        summary[portfolio][key] = count
    return summary


def _require_classified(con: duckdb.DuckDBPyConnection) -> None:
    """Reject incomplete classification before it hides evidence."""
    if (
        con.execute(
            "SELECT 1 FROM risk_factors WHERE ontology_status IS NULL "
            "OR ontology_status NOT IN ('resolved', 'unresolved') "
            "OR (ontology_status='resolved' AND "
            "(is_severe IS NULL OR ontology_level IS NULL OR risk_type IS NULL)) LIMIT 1"
        ).fetchone()
        is not None
    ):
        raise UnclassifiedRiskFactors(
            "Run classify_risk_factors before querying convergence evidence"
        )


def _support_query(
    con: duckdb.DuckDBPyConnection, filename: str, parameters: dict[str, object]
) -> pd.DataFrame:
    """Query one portfolio while preserving nullable evidence."""
    _require_classified(con)
    directory = Path(__file__).with_name("sql")
    # The prelude defines the scoping macros each statement calls. CREATE OR REPLACE makes it safe
    # to prepend every time, and DuckDB returns the cursor of the final statement.
    query = directory.joinpath("_prelude.sql").read_text(encoding="utf-8") + directory.joinpath(
        filename
    ).read_text(encoding="utf-8")
    result = con.execute(query, parameters)
    rows = result.fetchall()
    # Use object dtype so missing values stay None instead of turning into float NaN.
    return pd.DataFrame(
        {
            column[0]: pd.Series([row[i] for row in rows], dtype=object)
            for i, column in enumerate(result.description)
        }
    )


def suppressed_hubs(
    con: duckdb.DuckDBPyConnection, hub_max_countries: int, portfolio: str, limit: int
) -> pd.DataFrame:
    """Return a bounded shared-hub sample before severity filtering."""
    if limit < 0:
        raise ValueError("Suppressed hub limit must be nonnegative")
    return _support_query(
        con,
        "suppressed_hubs.sql",
        {
            "portfolio": portfolio,
            "hub_max_countries": hub_max_countries,
            "limit": limit,
        },
    )


def convergence_funnel(
    con: duckdb.DuckDBPyConnection, hub_max_countries: int, portfolio: str
) -> dict[str, int]:
    """Count one scoped population through both selection filters."""
    frame = _support_query(
        con,
        "convergence_funnel.sql",
        {
            "portfolio": portfolio,
            "hub_max_countries": hub_max_countries,
        },
    )
    counts = {str(key): int(value) for key, value in frame.to_dict("records")[0].items()}
    if counts["shared_nodes"] != counts["suppressed_hubs"] + counts["after_hub_suppression"]:
        raise ValueError("Hub suppression funnel does not reconcile")
    if counts["after_hub_suppression"] != (
        counts["removed_by_severity_filter"] + counts["after_severity_filter"]
    ):
        raise ValueError("Severity funnel does not reconcile")
    return counts


# A country limit no entity can reach suppresses nothing, so the funnel's final stage becomes the
# whole qualifying population rather than one threshold's survivors.
UNREACHABLE_COUNTRY_LIMIT = 10**6


def sensitivity_sweep(
    con: duckdb.DuckDBPyConnection, portfolio: str, thresholds: Iterable[int]
) -> dict[str, dict[str, int]]:
    """Measure retained and excluded qualifying nodes by country limit.

    Subtracting a threshold's survivors from the unsuppressed qualifying population gives the
    number excluded by the country rule while still carrying a selected factor. Computing it
    this way needs no extra query shape, so the published funnel keeps its exact columns.
    """
    qualifying = convergence_funnel(con, UNREACHABLE_COUNTRY_LIMIT, portfolio)[
        "after_severity_filter"
    ]
    sweep: dict[str, dict[str, int]] = {}
    for limit in sorted(set(thresholds)):
        # Reuse the published funnel so the retained count can't drift from the selection steps.
        retained = convergence_funnel(con, limit, portfolio)["after_severity_filter"]
        sweep[str(limit)] = {
            "retained": retained,
            "excluded_qualifying": qualifying - retained,
        }
    return sweep

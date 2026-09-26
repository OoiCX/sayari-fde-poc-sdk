"""Prepare report views from Findings without I/O or recomputing analytics."""

import json
from dataclasses import dataclass, fields
from typing import Any, cast
from unicodedata import combining, east_asian_width

from sayari_poc.models import Findings, OntologyFactor

# Cap the diagram at two rows of suppliers so the labels stay readable. The template says when it is
# showing a sample because membership is over this cap.
MAX_EXHIBIT_SUPPLIERS = 8
# Wrapped rows sit on 300-unit columns centred on the connector at x=600. The boxes are 276 wide, so
# the centres a row can use - 150, 450, 750 and 1050 - keep 588-612 clear down the whole diagram for
# the connectors to drop through.
EXHIBIT_COLUMNS = 4
EXHIBIT_COLUMN_WIDTH = 300
# Leave enough space between rows for the upper row's bus line to turn without touching a box.
EXHIBIT_ROW_GAP = 64
COVERAGE_LABELS = {
    "assessed": "Fully assessed",
    "partial": "Partially assessed",
    "no_data": "No upstream data found",
    "error": "Retrieval issue",
    "not_attempted": "Not assessed",
}

# Show each state's meaning next to its label, because the short labels are easy to misread. "No
# upstream data found" and "Not assessed" both mean absence of evidence, and neither is a finding of
# no risk; the section says so directly beneath this list.
COVERAGE_DETAILS = {
    "assessed": (
        "Upstream evidence was found; Sayari did not flag the search as partial. "
        "The result limit counts supply chain paths, not entities. "
        "Entities along those paths can outnumber the path limit. "
        "A search can stop at the path limit without being flagged as partial."
    ),
    "partial": (
        "Some upstream evidence was retrieved, but the available information may be incomplete."
    ),
    "no_data": "The lookup completed successfully, but no upstream relationships were returned.",
    "error": "Upstream data could not be retrieved successfully.",
    "not_attempted": (
        "Upstream analysis was not performed because a sufficiently reliable supplier "
        "match was not available."
    ),
}

# Use Sayari's terms and name what was excluded, so a shorter queue isn't read as safer evidence.
FUNNEL_LABELS = {
    "contributing_suppliers": "Matched suppliers with upstream trade evidence",
    "distinct_upstream_entities": "Upstream entities identified",
    "shared_nodes": "Upstream entities shared by more than one supplier",
    "suppressed_hubs": "Excluded — associated with too many countries",
    "after_hub_suppression": "Shared entities within the country limit",
    "removed_by_severity_filter": "Excluded — no critical or high risk factor",
    "after_severity_filter": "Shared entities flagged for review",
}

# Being left out of the ranking is not evidence of safety. Thresholds come from the manifest's
# methodology section rather than being copied into display constants here.
FUNNEL_DETAILS = {
    "contributing_suppliers": (
        "Suppliers matched to a Sayari company record whose upstream trade search returned "
        "at least one entity."
    ),
    "distinct_upstream_entities": (
        "Every distinct entity reached by the upstream trade search within the configured "
        "depth and result limit. A lower bound, not the whole network."
    ),
    "shared_nodes": (
        "Entities appearing upstream of at least two matched suppliers on this list. The "
        "connection is trade-record evidence, not proof of a direct supply relationship."
    ),
    "suppressed_hubs": (
        "Entities above the country limit stated in Methodology and limitations. These are "
        "typically logistics providers, trading intermediaries or large distributors, which "
        "would crowd the ranking without distinguishing between suppliers. Excluded from "
        "ranking only."
    ),
    "after_hub_suppression": "The shared entities carried into the risk factor rules.",
    "removed_by_severity_filter": (
        "Entities with no Sayari risk factor published at critical or high level. This is "
        "not a finding of no risk."
    ),
    "after_severity_filter": (
        "Entities meeting both the country limit and the risk factor rules. These are the "
        "entities listed under Shared upstream entities flagged for review."
    ),
}

# Leave the stored diagnostic reasons as they are; these display notes sit alongside them and
# explain the evidence gap and a sensible next step.
EXCEPTION_NOTES = {
    "No matching entity; profile not requested": (
        "No sufficiently reliable entity match was identified. Additional supplier details "
        "may be required to continue the analysis."
    ),
    "Weak match: needs adjudication; profile not requested": (
        "Possible entity match identified, but confidence was not high enough for automated "
        "analysis. Manual review recommended."
    ),
    "Resolution failed; profile not requested": (
        "The entity match could not be completed during this run. Retry or review the "
        "supplier manually."
    ),
    "Upstream retrieval failed; bounded evidence unavailable": (
        "Upstream information could not be retrieved during this run. Retry or review the "
        "supplier manually."
    ),
}

# A failed profile leaves risk unknown, so explain it by stage instead of treating missing evidence
# as absence of risk.
PROFILE_EXCEPTION_NOTE = (
    "Detailed profile information was unavailable for this entity, so its risk factors are "
    "unknown rather than absent. Other available evidence may still have been analyzed."
)


def exception_note(exception: dict[str, str]) -> str:
    """Explain a recorded gap or retain its unmapped reason.

    Ingestion problems and malformed-candidate diagnostics carry their own specific wording and
    are returned unchanged: inventing a general sentence for them would describe a row the run
    did not observe.
    """
    if exception.get("reason") in EXCEPTION_NOTES:
        return EXCEPTION_NOTES[exception["reason"]]
    if exception.get("stage") == "profile":
        return PROFILE_EXCEPTION_NOTE
    return exception.get("reason", "")


def supplier_list_label(portfolio: str | None, headline: str | None) -> str:
    """Name a portfolio without inventing a missing selection."""
    if portfolio is None:
        return "No supplier list selected"
    return "Screened supplier list" if portfolio == headline else f"Supplier list: {portfolio}"


def risk_categories(slugs: list[str], glossary: dict[str, OntologyFactor]) -> list[str]:
    """List the published categories of these factors, sorted and without repeats."""
    # A factor missing from the ontology has no published category, so it adds nothing.
    return sorted(
        {category for slug in slugs if slug in glossary for category in glossary[slug].categories}
    )


@dataclass(frozen=True)
class ExecutiveSummary:
    """Display counts retain their source grain and evidence bounds."""

    suppliers_screened: int
    resolved_rows: int
    resolved_entities: int
    unresolved_rows: int
    qualifying_nodes: int
    coverage_with_evidence: int
    coverage_denominator: int
    coverage_assessed: int
    coverage_partial: int


@dataclass(frozen=True)
class KeyFinding:
    """The first ranked headline node and its optional diagram."""

    node: dict[str, Any]
    exhibit: dict[str, Any] | None


@dataclass(frozen=True)
class PathHop:
    """One Sayari-annotated hop, named from Findings.paths.entities."""

    # Show Sayari's tier annotation as stored; don't derive a distance measure of our own.
    tier: int
    entity_id: str
    label: str | None
    translated_label: str | None
    components: list[dict[str, Any]]


@dataclass(frozen=True)
class PathRoute:
    """One distinct stored path record for one (node, supplier) pair."""

    hops: list[PathHop]
    occurrences: int
    # Record whether Sayari's path ran past this entity, so the caption can say so.
    truncated: bool = False


@dataclass(frozen=True)
class ExposureNode:
    """One ranked shared node this supplier's canonical entity sits under."""

    node: dict[str, Any]
    paths_available: bool
    path_observed: bool
    routes: list[PathRoute]
    retrieved_path_count: int
    stored_path_count: int
    # Connected suppliers with no retrieved path. That doesn't mean they have no route.
    suppliers_without_path_evidence: int


@dataclass(frozen=True)
class AdjudicationCandidate:
    """A valid candidate with its original response position."""

    response_index: int
    candidate: dict[str, Any]


@dataclass(frozen=True)
class SupplierRow:
    """One retained workbook row, independent of canonical duplicates."""

    portfolio: str
    row_number: int
    dom_id: str
    supplier: dict[str, object]
    candidate_rows: list[AdjudicationCandidate]
    exposure_state: str
    # None when exposure wasn't assessed, so the report shows unknown rather than a misleading 0.
    exposure_count: int | None
    exposure_denominator: int
    exposure_note: str
    exposure_nodes: list[ExposureNode]
    path_stored_count: int
    path_nodes_with_evidence: int
    path_nodes_without_evidence: int


@dataclass(frozen=True)
class ReportView:
    """Explicit template inputs, without exposing the full Findings object."""

    generated_at: str
    coverage: dict[str, dict[str, int]]
    exceptions: list[dict[str, str]]
    convergence: dict[str, dict[str, object]]
    manifest: dict[str, object]
    headline: str | None
    executive: ExecutiveSummary
    key_finding: KeyFinding | None
    glossary: dict[str, OntologyFactor]
    coverage_labels: dict[str, str]
    coverage_details: dict[str, str]
    funnel_labels: dict[str, str]
    funnel_details: dict[str, str]
    ranked_groups: list[dict[str, Any]]
    convergence_portfolios: list[str]
    supplier_rows: list[SupplierRow]
    # Dropdown choices for the two filters. Each lists only values some card carries, so no
    # choice can lead to an empty list on its own.
    supplier_country_codes: list[str]
    entity_country_codes: list[str]
    evidence_options: list[tuple[str, str]]
    category_options: list[tuple[str, str]]

    def context(self) -> dict[str, Any]:
        """The view is the template's only evidence boundary."""
        # Build the context from the dataclass fields only, so nothing undeclared can leak through.
        return {field.name: getattr(self, field.name) for field in fields(self)}


def headline_portfolio(findings: Findings) -> str | None:
    """Read the pipeline's headline selection without replacing it."""
    value = findings.manifest.get("headline_portfolio")
    return str(value) if value is not None else None


def _label_lines(label: str, width: int) -> list[str]:
    """Wrap words using display widths for CJK glyphs and accents."""
    lines: list[str] = []
    for paragraph in label.splitlines():
        remaining = paragraph.strip()
        while remaining:
            cells, end = 0, 0
            for char in remaining:
                # Combining marks take no cells, wide/fullwidth glyphs take two, the rest one.
                size = 0 if combining(char) else 2 if east_asian_width(char) in {"W", "F"} else 1
                if cells and cells + size > width:
                    break
                cells += size
                end += 1
            # Break at the last space if the width cutoff would split a word.
            if end < len(remaining) and not remaining[end].isspace():
                boundary = remaining.rfind(" ", 0, end + 1)
                if boundary > 0:
                    end = boundary
            lines.append(remaining[:end].rstrip())
            remaining = remaining[end:].lstrip()
    return lines or [""]


def _exhibit(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Lay out valid membership or return None for an unusable diagram."""
    if not node or not isinstance(node.get("label"), str) or not node["label"].strip():
        return None
    suppliers = node.get("suppliers", [])
    if (
        not isinstance(suppliers, list)
        or len(suppliers) != node.get("supplier_count")
        or len(suppliers) < 2
        or any(
            not s.get("entity_id") or not isinstance(s.get("label"), str) or not s["label"].strip()
            for s in suppliers
        )
    ):
        # Skip the diagram rather than draw incomplete or inconsistent supplier membership.
        return None
    # Duplicate canonical suppliers would make convergence look bigger than it is.
    if len({s["entity_id"] for s in suppliers}) != len(suppliers):
        return None
    selected = sorted(suppliers, key=lambda s: s["entity_id"])[:MAX_EXHIBIT_SUPPLIERS]
    drawn = []
    for supplier in selected:
        lines = _label_lines(supplier["label"], 30)
        if supplier.get("translated_label"):
            lines += _label_lines(supplier["translated_label"], 30)
        drawn.append({**supplier, "lines": lines})
    # Make every supplier box tall enough for the longest selected label, translation included.
    box_height = 52 + 22 * max(len(s["lines"]) for s in drawn)
    per_row = len(drawn) if len(drawn) <= EXHIBIT_COLUMNS else -(-len(drawn) // 2)
    rows = [drawn[start : start + per_row] for start in range(0, len(drawn), per_row)]
    odd_rows = 0
    for row_index, row in enumerate(rows):
        box_y = 36 + row_index * (box_height + EXHIBIT_ROW_GAP)
        # A row with an odd count can't straddle x=600 without putting a box on the connector, so
        # shift it half a column. Alternating the direction keeps a two-row block centred overall.
        shift = 0.0 if len(row) % 2 == 0 else EXHIBIT_COLUMN_WIDTH / 2 * (-1) ** (odd_rows + 1)
        odd_rows += len(row) % 2
        for column, supplier in enumerate(row):
            # Spread a single row across the canvas. Centre wrapped rows on the connector at x=600
            # instead, so the supplier block sits over the shared entity box rather than to its
            # left. Every centre stays at least 150 from 600, which keeps the 588-612 corridor
            # clear: connectors from an upper row drop through it instead of across a lower box.
            supplier["x"] = (
                (column + 0.5) * 1200 / per_row
                if len(rows) == 1
                else 600 + EXHIBIT_COLUMN_WIDTH * (column - (len(row) - 1) / 2) + shift
            )
            supplier["box_y"] = box_y
            supplier["box_bottom"] = box_y + box_height
            supplier["bus_y"] = box_y + box_height + 24
            supplier["text_y"] = box_y + (box_height - 22 * (len(supplier["lines"]) - 1)) / 2 + 5
    node_y = 36 + len(rows) * box_height + (len(rows) - 1) * EXHIBIT_ROW_GAP + 116
    lines = _label_lines(node["label"], 72)
    if node.get("translated_label"):
        lines += _label_lines(node["translated_label"], 72)
    return {
        "node": node,
        "suppliers": drawn,
        "box_height": box_height,
        "bus_lines": sorted({s["bus_y"] for s in drawn}),
        "node_y": node_y,
        "edge_label_y": node_y - 24,
        "node_lines": lines,
        "height": node_y + 98 + 22 * len(lines),
    }


# Trim at the displayed entity before grouping; otherwise paths that differ only further upstream
# would show up as examples that look identical.
def _hops_to_entity(hops: list[dict[str, Any]], node_id: str) -> list[dict[str, Any]]:
    """Trim a path at the displayed entity when it is present."""
    for index, hop in enumerate(hops):
        if hop["entity_id"] == node_id:
            return hops[: index + 1]
    # A route that never reaches the entity is left exactly as received.
    return hops


def _path_routes(
    records: list[dict[str, Any]], entity_id: str, labels: dict[str, Any], node_id: str
) -> list[PathRoute]:
    """Group identical prefixes while preserving occurrence counts."""
    # The order of components and country arrays still counts when deciding two paths are equal.
    groups: dict[str, list[tuple[list[dict[str, Any]], bool]]] = {}
    for record in records:
        if record["source_entity_id"] == entity_id:
            # Trim before grouping, so routes that share a prefix collapse into one.
            hops = _hops_to_entity(record["hops"], node_id)
            key = json.dumps(hops, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            groups.setdefault(key, []).append((hops, len(hops) < len(record["hops"])))
    return [
        PathRoute(
            hops=[
                PathHop(
                    tier=hop["tier"],
                    entity_id=hop["entity_id"],
                    label=labels[hop["entity_id"]]["label"],
                    translated_label=labels[hop["entity_id"]].get("translated_label"),
                    components=hop["components"],
                )
                for hop in group[0][0]
            ],
            occurrences=len(group),
            truncated=any(truncated for _, truncated in group),
        )
        for group in groups.values()
    ]


def _exposure_node(findings: Findings, node: dict[str, Any], entity_id: str) -> ExposureNode:
    """Attach headline-scoped path evidence to one supplier finding."""
    # The same upstream ID can appear in another portfolio, but paths only cover the headline.
    paths: Any = findings.paths
    record = (
        paths.get("nodes", {}).get(node["upstream_id"])
        if node["portfolio"] == headline_portfolio(findings)
        else None
    )
    observed = record["path_observed_supplier_ids"] if record is not None else []
    # When paths_available is false the zero counts are placeholders, not evidence of no route.
    return ExposureNode(
        node=node,
        paths_available=record is not None,
        path_observed=entity_id in observed,
        routes=_path_routes(record["paths"], entity_id, paths["entities"], node["upstream_id"])
        if record is not None
        else [],
        retrieved_path_count=record["retrieved_path_count"] if record is not None else 0,
        stored_path_count=record["stored_path_count"] if record is not None else 0,
        suppliers_without_path_evidence=node["supplier_count"] - len(observed)
        if record is not None
        else 0,
    )


def _candidate_rows(supplier: dict[str, object]) -> list[AdjudicationCandidate]:
    """Recover candidate positions across malformed-response gaps."""
    errors = {int(index) for index in cast(dict[str, str], supplier.get("candidate_errors", {}))}
    candidates = cast(list[dict[str, Any]], supplier.get("candidates", []))
    rows = []
    position = 0
    # Give each valid candidate its original response position; don't re-rank the list.
    for candidate in candidates:
        while position in errors:
            position += 1
        rows.append(AdjudicationCandidate(response_index=position, candidate=candidate))
        position += 1
    return rows


def _supplier_rows(findings: Findings) -> list[SupplierRow]:
    """Build one investigation view per input row."""
    rows = []
    canonical = sorted(
        findings.suppliers,
        key=lambda supplier: (cast(str, supplier["portfolio"]), cast(int, supplier["row_number"])),
    )
    notes = {
        "no_data": "Not assessed — upstream retrieval returned no evidence for this supplier.",
        "error": "Not assessed — upstream retrieval failed for this supplier.",
        "not_attempted": "Not assessed — upstream retrieval was not attempted for this supplier.",
    }
    for position, supplier in enumerate(canonical, 1):
        portfolio = cast(str, supplier["portfolio"])
        nodes: list[dict[str, Any]] = [
            node for node in findings.shared_nodes if node["portfolio"] == portfolio
        ]
        entity_id = cast(str | None, supplier.get("entity_id"))
        accepted = supplier.get("resolution_status") == "resolved" and entity_id is not None
        coverage = cast(str, supplier.get("coverage_status") or "not_attempted")
        if not accepted:
            # Weak or failed matches stay not_attempted even if another row has their candidate ID.
            coverage = "not_attempted"
        exposed = [
            _exposure_node(findings, node, entity_id)
            for node in nodes
            if accepted
            and entity_id is not None
            and coverage in {"assessed", "partial"}
            and any(member["entity_id"] == entity_id for member in node["suppliers"])
        ]
        # Partial (bounded) evidence counts as assessed too; `coverage` still records which it was.
        assessed = accepted and coverage in {"assessed", "partial"}
        rows.append(
            SupplierRow(
                portfolio=portfolio,
                row_number=cast(int, supplier["row_number"]),
                dom_id=f"supplier-{position}",
                supplier=supplier,
                candidate_rows=_candidate_rows(supplier),
                exposure_state="exposed"
                if exposed
                else "none_observed"
                if assessed
                else "not_assessed",
                exposure_count=len(exposed) if assessed else None,
                exposure_denominator=len(nodes),
                exposure_note="" if assessed else notes.get(coverage, notes["not_attempted"]),
                exposure_nodes=exposed,
                path_stored_count=sum(
                    route.occurrences for node in exposed for route in node.routes
                ),
                path_nodes_with_evidence=sum(node.path_observed for node in exposed),
                path_nodes_without_evidence=sum(not node.path_observed for node in exposed),
            )
        )
    # Most exposed suppliers first; break ties by portfolio and row number so the order is stable.
    return sorted(rows, key=lambda row: (-(row.exposure_count or 0), row.portfolio, row.row_number))


def build_report_view(findings: Findings) -> ReportView:
    """Project Findings into bounded, offline report inputs."""
    headline = headline_portfolio(findings)
    nodes: list[dict[str, Any]] = [
        node for node in findings.shared_nodes if node.get("portfolio") == headline
    ]
    # Grouping is for display only; nodes within each portfolio keep the order analysis gave them.
    portfolios = sorted(
        {str(node["portfolio"]) for node in findings.shared_nodes},
        key=lambda portfolio: (portfolio != headline, portfolio),
    )
    ranked_groups = [
        {
            "portfolio": portfolio,
            "nodes": [node for node in findings.shared_nodes if node["portfolio"] == portfolio],
        }
        for portfolio in portfolios
    ]
    convergence_portfolios = sorted(
        findings.convergence,
        key=lambda portfolio: (portfolio != headline, portfolio),
    )
    headline_rows = [s for s in findings.suppliers if s.get("portfolio") == headline]
    resolved_rows = [row for row in headline_rows if row.get("resolution_status") == "resolved"]
    coverage = findings.coverage.get(headline or "", {})
    executive = ExecutiveSummary(
        suppliers_screened=len(headline_rows),
        resolved_rows=len(resolved_rows),
        resolved_entities=len({row["entity_id"] for row in resolved_rows if row.get("entity_id")}),
        unresolved_rows=len(headline_rows) - len(resolved_rows),
        qualifying_nodes=len(nodes),
        coverage_with_evidence=coverage.get("assessed", 0) + coverage.get("partial", 0),
        coverage_denominator=sum(coverage.get(state, 0) for state in COVERAGE_LABELS),
        coverage_assessed=coverage.get("assessed", 0),
        coverage_partial=coverage.get("partial", 0),
    )
    # Read each supplier's state the same way its card displays it, so the filter and the card
    # always agree.
    states = {str(row.get("coverage_status") or "not_attempted") for row in findings.suppliers}
    # Categories come from the critical and high factors only, because those are why an entity
    # is flagged. The label is Sayari's own category name with its underscores removed.
    categories = {
        category
        for node in findings.shared_nodes
        for category in risk_categories(cast(list[str], node["severe_factors"]), findings.ontology)
    }
    return ReportView(
        generated_at=findings.generated_at,
        coverage=findings.coverage,
        exceptions=findings.exceptions,
        convergence=findings.convergence,
        manifest=findings.manifest,
        headline=headline,
        executive=executive,
        key_finding=KeyFinding(node=nodes[0], exhibit=_exhibit(nodes[0])) if nodes else None,
        glossary=findings.ontology,
        coverage_labels=COVERAGE_LABELS,
        coverage_details=COVERAGE_DETAILS,
        funnel_labels=FUNNEL_LABELS,
        funnel_details=FUNNEL_DETAILS,
        ranked_groups=ranked_groups,
        convergence_portfolios=convergence_portfolios,
        supplier_rows=_supplier_rows(findings),
        supplier_country_codes=sorted(
            {str(row["input_country"]) for row in findings.suppliers if row.get("input_country")}
        ),
        entity_country_codes=sorted(
            {
                code
                for node in findings.shared_nodes
                for code in cast(list[str], node.get("countries") or [])
            }
        ),
        evidence_options=[
            (state, label) for state, label in COVERAGE_LABELS.items() if state in states
        ],
        category_options=[
            (category, category.replace("_", " ").capitalize()) for category in sorted(categories)
        ],
    )

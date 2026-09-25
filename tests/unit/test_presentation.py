"""Presentation prepares an immutable view without rewriting evidence or analytics."""

import ast
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from sayari_poc import presentation
from sayari_poc.models import Findings
from sayari_poc.presentation import ReportView, build_report_view, headline_portfolio


def test_build_is_deterministic_preserves_findings_and_has_exact_context(
    findings: Findings,
) -> None:
    # Building the view leaves Findings unchanged and exposes only the declared fields.
    before = findings.model_dump()
    view = build_report_view(findings)
    assert view == build_report_view(findings)
    assert findings.model_dump() == before
    assert isinstance(view, ReportView) and is_dataclass(view)
    with pytest.raises(FrozenInstanceError):
        cast(Any, view).headline = "changed"
    context = view.context()
    assert set(context) == {
        "generated_at",
        "coverage",
        "exceptions",
        "convergence",
        "manifest",
        "headline",
        "executive",
        "key_finding",
        "glossary",
        "coverage_labels",
        "coverage_details",
        "funnel_labels",
        "funnel_details",
        "ranked_groups",
        "convergence_portfolios",
        "supplier_rows",
    }
    assert len(context) == 16
    assert all(not isinstance(value, Findings) for value in context.values())
    assert context["generated_at"] == findings.generated_at
    assert context["coverage"] == findings.coverage
    assert context["exceptions"] == findings.exceptions
    assert context["convergence"] == findings.convergence
    assert context["manifest"] == findings.manifest
    assert (
        view.headline,
        view.executive.suppliers_screened,
        view.executive.resolved_rows,
        view.executive.qualifying_nodes,
    ) == (
        "list_3",
        8,
        7,
        2,
    )
    assert view.glossary == findings.ontology
    assert view.coverage_labels == presentation.COVERAGE_LABELS
    # Pin the wording independently of the production mapping: a failed or unattempted retrieval
    # must never read as a successful empty one (an honesty rule).
    assert presentation.COVERAGE_LABELS == {
        "assessed": "Fully assessed",
        "partial": "Partially assessed",
        "no_data": "No upstream data found",
        "error": "Retrieval issue",
        "not_attempted": "Not assessed",
    }
    assert len(set(presentation.COVERAGE_LABELS.values())) == 5
    assert view.funnel_labels == presentation.FUNNEL_LABELS
    assert view.funnel_details == presentation.FUNNEL_DETAILS
    assert set(view.funnel_details) == set(view.funnel_labels)


def test_raw_evidence_and_ranked_order_are_preserved_verbatim(findings: Findings) -> None:
    # The view shows source evidence verbatim and keeps the producer's ranking within each group.
    top = next(
        node
        for node in cast(list[dict[str, Any]], findings.shared_nodes)
        if node["portfolio"] == "list_3"
    )
    top.update(
        upstream_id="Canonical/原始-ID",
        label="石家庄泰明顿摩擦材料有限公司",
        translated_label="Literal <tag> & translation",
        severe_factors=["unknown_RAW_factor", "psa_sanctioned_synthetic"],
        other_factors=["unmapped_slug"],
    )
    before = findings.model_dump()
    view = build_report_view(findings)
    assert [group["portfolio"] for group in view.ranked_groups] == ["list_3", "another_portfolio"]
    assert view.convergence_portfolios == ["list_3", "another_portfolio"]
    for group in view.ranked_groups:
        assert group["nodes"] == [
            node
            for node in cast(list[dict[str, Any]], findings.shared_nodes)
            if node["portfolio"] == group["portfolio"]
        ]
    assert view.key_finding is not None
    assert view.key_finding.exhibit is not None
    assert view.ranked_groups[0]["nodes"][0] == top == view.key_finding.exhibit["node"]
    assert view.key_finding.exhibit["node_lines"] == [top["label"], top["translated_label"]]
    assert findings.model_dump() == before


def test_exhibit_selects_only_the_first_ranked_node(findings: Findings) -> None:
    # The diagram shows the first ranked node and never picks a different one.
    top = next(
        node
        for node in cast(list[dict[str, Any]], findings.shared_nodes)
        if node["portfolio"] == "list_3"
    )
    top["suppliers"].reverse()
    before = findings.model_dump()
    view = build_report_view(findings)
    assert view.key_finding is not None
    first = view.key_finding.exhibit
    assert first is not None
    assert first["node"] == top
    # The node is within the supplier cap, so every connected supplier is drawn.
    assert [supplier["entity_id"] for supplier in first["suppliers"]] == [
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ]
    assert presentation.MAX_EXHIBIT_SUPPLIERS == 8
    assert len(top["suppliers"]) == 6
    # Six suppliers wrap onto two rows of three, and no box straddles the connector corridor at
    # x=600 that the upper row's connectors descend through.
    rows = {supplier["box_y"] for supplier in first["suppliers"]}
    assert rows == {36, 36 + first["box_height"] + 64}
    assert all(abs(supplier["x"] - 600) >= 138 for supplier in first["suppliers"])
    # The two rows are staggered half a column in opposite directions, which centres the drawn block
    # on the same x=600 axis as the shared entity box beneath it.
    placed = sorted((supplier["box_y"], supplier["x"]) for supplier in first["suppliers"])
    assert [x for _, x in placed[:3]] == [150, 450, 750]
    assert [x for _, x in placed[3:]] == [450, 750, 1050]
    horizontal = [x for _, x in placed]
    assert (min(horizontal) + max(horizontal)) / 2 == 600
    assert findings.model_dump() == before
    # An invalid first node gets no diagram, and a lower-ranked node never takes its place.
    top["label"] = None
    rejected = build_report_view(findings)
    assert rejected.key_finding is not None
    assert rejected.key_finding.exhibit is None


@pytest.mark.parametrize("missing", ["nodes", "headline"])
def test_missing_exhibit_sources(findings: Findings, missing: str) -> None:
    # Without headline evidence the view draws no diagram and invents no summary counts.
    if missing == "nodes":
        findings.shared_nodes = []
    if missing == "headline":
        cast(dict[str, Any], findings.manifest).pop("headline_portfolio")
    view = build_report_view(findings)
    assert (view.key_finding.exhibit if view.key_finding else None) is None
    if missing == "headline":
        assert view.headline is None
        assert (
            view.executive.suppliers_screened,
            view.executive.resolved_rows,
            view.executive.qualifying_nodes,
        ) == (0, 0, 0)


@pytest.fixture
def exhibit_node() -> dict[str, Any]:
    return {
        "upstream_id": "canonical-node",
        "label": "原始节点",
        "supplier_count": 2,
        "suppliers": [
            {"entity_id": "b", "label": "供应商乙"},
            {"entity_id": "a", "label": "供应商甲"},
        ],
    }


@pytest.mark.parametrize("invalid", [None, {}, {"label": None}, {"label": 1}, {"label": " 	"}])
def test_exhibit_node_none_guards(invalid: dict[str, Any] | None) -> None:
    # A shared node without a usable label gets no diagram.
    assert presentation._exhibit(invalid) is None


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "not_list",
        "count_mismatch",
        "empty",
        "one",
        "missing_id",
        "null_id",
        "blank_id",
        "missing_label",
        "null_label",
        "non_string_label",
        "blank_label",
        "duplicate_id",
    ],
)
def test_exhibit_supplier_none_guards(exhibit_node: dict[str, Any], change: str) -> None:
    # Incomplete or inconsistent supplier data means no diagram, and the input is left unchanged.
    suppliers = exhibit_node["suppliers"]
    if change == "missing":
        del exhibit_node["suppliers"]
    elif change == "not_list":
        exhibit_node["suppliers"] = None
    elif change == "count_mismatch":
        exhibit_node["supplier_count"] = 3
    elif change in {"empty", "one"}:
        exhibit_node["suppliers"] = suppliers[: 0 if change == "empty" else 1]
        exhibit_node["supplier_count"] = len(exhibit_node["suppliers"])
    elif change == "duplicate_id":
        suppliers[1]["entity_id"] = suppliers[0]["entity_id"]
    elif change == "missing_id":
        del suppliers[0]["entity_id"]
    elif change == "missing_label":
        del suppliers[0]["label"]
    else:
        key, value = {
            "null_id": ("entity_id", None),
            "blank_id": ("entity_id", ""),
            "null_label": ("label", None),
            "non_string_label": ("label", 1),
            "blank_label": ("label", " 	"),
        }[change]
        suppliers[0][key] = value
    before = deepcopy(exhibit_node)
    assert presentation._exhibit(exhibit_node) is None
    assert exhibit_node == before


@pytest.mark.parametrize(
    "label, width, expected",
    [
        (
            "ADIENT US ENTERPRISES LIMITED PARTNERSHIP",
            30,
            ["ADIENT US ENTERPRISES LIMITED", "PARTNERSHIP"],
        ),
        ("Long Company Name", 10, ["Long", "Company", "Name"]),
        ("ABCDEFGHIJK", 5, ["ABCDE", "FGHIJ", "K"]),
        ("供应商甲供应商乙", 8, ["供应商甲", "供应商乙"]),
        ("Cafe\u0301 Group", 4, ["Cafe\u0301", "Grou", "p"]),
        ("Original\r\nTranslation", 30, ["Original", "Translation"]),
    ],
)
def test_diagram_labels_wrap_words_and_retain_unicode(
    label: str, width: int, expected: list[str]
) -> None:
    # Labels wrap at word boundaries and measure Unicode text by its display width.
    assert presentation._label_lines(label, width) == expected


def test_valid_exhibit_without_translations_preserves_source(exhibit_node: dict[str, Any]) -> None:
    # Native-language labels are used as they are when there is no translation.
    before = deepcopy(exhibit_node)
    result = presentation._exhibit(exhibit_node)
    assert result is not None
    assert [supplier["entity_id"] for supplier in result["suppliers"]] == ["a", "b"]
    assert result["node_lines"] == ["原始节点"]
    assert [supplier["lines"] for supplier in result["suppliers"]] == [["供应商甲"], ["供应商乙"]]
    assert exhibit_node == before


@pytest.mark.parametrize("value, expected", [(None, None), ("custom", "custom"), (123, "123")])
def test_headline_reads_only_supplied_manifest(value: object, expected: str | None) -> None:
    # The headline portfolio comes only from the manifest; the view never invents one.
    findings = Findings(
        generated_at="injected", suppliers=[], manifest={"headline_portfolio": value}
    )
    assert headline_portfolio(findings) == expected


def test_presentation_imports_only_models_and_stdlib() -> None:
    # The view layer imports only models and the standard library, so it cannot reach retrieval, the
    # warehouse or template I/O.
    tree = ast.parse(Path(presentation.__file__).read_text(encoding="utf-8"))
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
    assert all(
        name == "sayari_poc.models" or cast(str, name).split(".")[0] in sys.stdlib_module_names
        for name in imports
    )


def test_executive_summary_fields_match_hand_computed_evidence(findings: Findings) -> None:
    # The executive totals match row and entity counts worked out by hand from the fixture.
    before = findings.model_dump()
    executive = build_report_view(findings).executive
    assert vars(executive) == {
        "suppliers_screened": 8,
        "resolved_rows": 7,
        "resolved_entities": 6,
        "unresolved_rows": 1,
        "qualifying_nodes": 2,
        "coverage_with_evidence": 7,
        "coverage_denominator": 8,
        "coverage_assessed": 6,
        "coverage_partial": 1,
    }
    assert findings.model_dump() == before


def test_key_finding_follows_first_headline_node(findings: Findings) -> None:
    # The key finding is the headline portfolio's first-ranked node, exactly as ranked.
    cast(list[dict[str, Any]], findings.shared_nodes).reverse()
    before = findings.model_dump()
    expected = next(
        node
        for node in cast(list[dict[str, Any]], findings.shared_nodes)
        if node["portfolio"] == "list_3"
    )
    key = build_report_view(findings).key_finding
    assert key is not None and key.node is expected
    assert key.exhibit is not None and key.exhibit["node"] is expected
    assert findings.model_dump() == before


def test_key_finding_absent_when_only_other_portfolio_has_nodes(findings: Findings) -> None:
    # A node from another portfolio can never become the headline's key finding.
    findings.shared_nodes = [
        node
        for node in cast(list[dict[str, Any]], findings.shared_nodes)
        if node["portfolio"] != "list_3"
    ]
    view = build_report_view(findings)
    assert view.key_finding is None
    assert view.executive.qualifying_nodes == 0
    assert view.ranked_groups  # Other portfolios remain available in the evidence table.


def test_key_finding_keeps_evidence_when_exhibit_is_omitted(findings: Findings) -> None:
    # Dropping the diagram does not drop the ranked finding behind it.
    top = next(
        node
        for node in cast(list[dict[str, Any]], findings.shared_nodes)
        if node["portfolio"] == "list_3"
    )
    top["label"] = None
    key = build_report_view(findings).key_finding
    assert key is not None and key.node is top
    assert key.exhibit is None


def test_executive_and_key_finding_are_frozen_dataclasses(findings: Findings) -> None:
    # The executive and key-finding records are frozen, so templates cannot change them.
    view = build_report_view(findings)
    assert isinstance(view.executive, presentation.ExecutiveSummary)
    assert isinstance(view.key_finding, presentation.KeyFinding)
    with pytest.raises(FrozenInstanceError):
        cast(Any, view.executive).suppliers_screened = 999
    with pytest.raises(FrozenInstanceError):
        cast(Any, view.key_finding).node = {}


def test_supplier_rows_preserve_fifty_input_keys_and_canonical_duplicates(
    findings: Findings,
) -> None:
    # All 50 input rows keep their own anchor, even when they share one canonical entity.
    source = next(
        row for row in cast(list[dict[str, Any]], findings.suppliers) if row["entity_id"] == "s1"
    )
    findings.suppliers = [
        {**source, "row_number": number, "input_name": f"Input <{number}> 原始"}
        for number in range(2, 52)
    ]
    before = findings.model_dump()
    rows = build_report_view(findings).supplier_rows
    assert len(rows) == 50
    assert [(row.portfolio, row.row_number) for row in rows] == [
        (source["portfolio"], number) for number in range(2, 52)
    ]
    assert [row.dom_id for row in rows] == [f"supplier-{number}" for number in range(1, 51)]
    assert len({row.exposure_count for row in rows}) == 1
    assert rows == build_report_view(findings).supplier_rows
    assert findings.model_dump() == before


def test_supplier_exposure_is_independent_portfolio_membership_and_input_order(
    findings: Findings,
) -> None:
    # Exposure is counted within each portfolio, and duplicate inputs stay distinct. The ranks and
    # memberships differ, so neither the fixture's counts nor its entity order can pass by accident.
    cast(list[dict[str, Any]], findings.shared_nodes).reverse()
    shared_nodes: Any = cast(list[dict[str, Any]], findings.shared_nodes)
    for index, node in enumerate(shared_nodes):
        node["suppliers"] = node["suppliers"][index % 2 :]
        node["supplier_count"] = len(node["suppliers"])
    cast(list[dict[str, Any]], findings.suppliers).reverse()
    before = findings.model_dump()
    rows = build_report_view(findings).supplier_rows
    canonical = sorted(
        cast(list[dict[str, Any]], findings.suppliers),
        key=lambda row: (row["portfolio"], row["row_number"]),
    )
    expected_order = []
    for row in rows:
        source = row.supplier
        nodes = [n for n in shared_nodes if n["portfolio"] == source["portfolio"]]
        matches = [
            n for n in nodes if source["entity_id"] in {s["entity_id"] for s in n["suppliers"]}
        ]
        expected = len(matches) if source["resolution_status"] == "resolved" else None
        assert row.exposure_count == expected
        assert row.exposure_denominator == len(nodes)
        assert [node.node for node in row.exposure_nodes] == matches
        assert row.dom_id == f"supplier-{canonical.index(source) + 1}"
        expected_order.append((-(expected or 0), row.portfolio, row.row_number))
    assert expected_order == sorted(expected_order)
    duplicate = [row for row in rows if row.supplier["entity_id"] == "s1"]
    assert len(duplicate) == 2
    assert duplicate[0].dom_id != duplicate[1].dom_id
    assert duplicate[0].exposure_count == duplicate[1].exposure_count
    assert rows == build_report_view(findings).supplier_rows
    assert findings.model_dump() == before


@pytest.mark.parametrize(
    "resolution, coverage, entity_id, expected_count, expected_state, note",
    [
        ("weak", "assessed", "s1", None, "not_assessed", "was not attempted"),
        ("no_match", "assessed", None, None, "not_assessed", "was not attempted"),
        ("error", "partial", "s1", None, "not_assessed", "was not attempted"),
        ("resolved", "no_data", "s1", None, "not_assessed", "returned no evidence"),
        ("resolved", "error", "s1", None, "not_assessed", "failed"),
        ("resolved", "not_attempted", "s1", None, "not_assessed", "was not attempted"),
        ("resolved", "assessed", "absent", 0, "none_observed", ""),
        ("resolved", "partial", "absent", 0, "none_observed", ""),
        ("resolved", "assessed", "s1", 2, "exposed", ""),
        ("resolved", "partial", "s1", 2, "exposed", ""),
    ],
)
def test_supplier_exposure_absence_never_invents_an_assessment(
    findings: Findings,
    resolution: str,
    coverage: str,
    entity_id: str | None,
    expected_count: int | None,
    expected_state: str,
    note: str,
) -> None:
    # When retrieval gave no usable evidence, exposure stays unknown; it is never shown as zero.
    source = next(
        row for row in cast(list[dict[str, Any]], findings.suppliers) if row["entity_id"] == "s1"
    )
    findings.suppliers = [
        {
            **source,
            "resolution_status": resolution,
            "coverage_status": coverage,
            "entity_id": entity_id,
        }
    ]
    row = build_report_view(findings).supplier_rows[0]
    assert row.exposure_state == expected_state
    assert row.exposure_count == expected_count
    assert row.exposure_note == (
        f"Not assessed — upstream retrieval {note} for this supplier." if note else ""
    )
    if expected_state != "exposed":
        assert not row.exposure_nodes
        assert row.path_stored_count == 0


def test_supplier_path_projection_keeps_node_grain_source_order_and_full_records(
    findings: Findings,
) -> None:
    # A supplier's path view keeps the per-node totals and the complete path records in order.
    components = [
        {
            "hs_code": "0099",
            "departure_countries": ["ZZZ", "AAA", "ZZZ"],
            "arrival_countries": [],
            "min_date": "2021-03-02",
            "max_date": "2025-06-19",
        },
        {
            "hs_code": "8542",
            "departure_countries": [],
            "arrival_countries": ["CHN", "USA"],
            "min_date": None,
            "max_date": None,
        },
    ]
    hops: list[dict[str, Any]] = [
        {"entity_id": "hop", "tier": 2, "components": components},
        {"entity_id": "n-top", "tier": 4, "components": components[::-1]},
    ]
    first: dict[str, Any] = {"source_entity_id": "s1", "path_index": 7, "hops": hops}
    other = deepcopy(first)
    other["hops"][0]["components"][0]["hs_code"] = "different-evidence"
    findings.paths = {
        "entities": {
            "hop": {"label": "中间节点", "translated_label": "Intermediate"},
            "n-top": {"label": "Named target", "translated_label": None},
        },
        "nodes": {
            "n-top": {
                "retrieved_path_count": 12,
                "stored_path_count": 6,
                "truncated": True,
                "path_observed_supplier_ids": ["s1"],
                "paths": [first, deepcopy(first), other, deepcopy(other), deepcopy(first), other],
            },
            "n-boundary": {
                "retrieved_path_count": 0,
                "stored_path_count": 0,
                "truncated": False,
                "path_observed_supplier_ids": [],
                "paths": [],
            },
        },
    }
    before = findings.model_dump()
    rows = build_report_view(findings).supplier_rows
    row = next(row for row in rows if row.supplier["entity_id"] == "s1")
    node = row.exposure_nodes[0]
    assert (node.retrieved_path_count, node.stored_path_count) == (12, 6)
    assert node.paths_available and node.path_observed
    assert node.suppliers_without_path_evidence == 5
    assert len(node.routes) == 2
    assert [route.occurrences for route in node.routes] == [3, 3]
    for route, original in zip(node.routes, [first, other], strict=True):
        assert [(hop.entity_id, hop.tier) for hop in route.hops] == [("hop", 2), ("n-top", 4)]
        assert [hop.components for hop in route.hops] == [
            hop["components"] for hop in original["hops"]
        ]
        assert route.hops[0].label == "中间节点"
        assert route.hops[0].translated_label == "Intermediate"
    assert (
        row.path_stored_count,
        row.path_nodes_with_evidence,
        row.path_nodes_without_evidence,
    ) == (6, 1, 1)
    missing = next(row for row in rows if row.supplier["entity_id"] == "s2").exposure_nodes[0]
    assert missing.paths_available and not missing.path_observed and not missing.routes
    assert missing.suppliers_without_path_evidence == 5
    other_portfolio = next(row for row in rows if row.portfolio == "another_portfolio")
    assert all(
        not node.paths_available and not node.routes for node in other_portfolio.exposure_nodes
    )
    for value in (row, node, node.routes[0], node.routes[0].hops[0]):
        assert is_dataclass(value)
        field = next(iter(vars(value)))
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, None)
    assert build_report_view(findings).supplier_rows == rows
    assert findings.model_dump() == before


@pytest.mark.parametrize(
    "portfolio, headline, expected",
    [
        ("list_3", "list_3", "Screened supplier list"),
        ("any_name", "any_name", "Screened supplier list"),
        ("list_3", "other", "Supplier list: list_3"),
        ("another_portfolio", "any_name", "Supplier list: another_portfolio"),
        (None, None, "No supplier list selected"),
    ],
)
def test_supplier_list_display_depends_on_role(
    portfolio: str | None, headline: str | None, expected: str
) -> None:
    # The supplier-list label marks the headline portfolio differently from other sheet names.
    assert presentation.supplier_list_label(portfolio, headline) == expected

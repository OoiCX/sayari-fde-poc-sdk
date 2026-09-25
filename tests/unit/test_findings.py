"""Pure Findings assembly preserves stage evidence and input-row accounting."""

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from sayari_poc import findings as assembly
from sayari_poc.config import Settings
from sayari_poc.findings import assemble_findings
from sayari_poc.models import (
    EntityProfile,
    IngestResult,
    InputEntity,
    ResolvedEntity,
    UpstreamResult,
)
from tests.ontology_support import synthetic_ontology


def test_error_without_message_never_gets_no_match_explanation() -> None:
    # A resolution failure with no message still never reads as "no match".
    row = ResolvedEntity(sheet="portfolio", row_number=2, input_name="Synthetic", status="error")
    diagnostics = assembly._row_exceptions(row)
    assert diagnostics[0]["reason"] == "Resolution failed; profile not requested"
    assert diagnostics[0]["error_type"] == "error"


def test_candidate_diagnostics_sort_original_indices_numerically() -> None:
    # Candidate diagnostics sort by response position as numbers, so 2 comes before 10.
    row = ResolvedEntity(sheet="portfolio", row_number=2, input_name="Synthetic", status="resolved")
    row.candidate_errors = {10: "Malformed alternate", 2: "Malformed alternate", 1: "Invalid"}
    first = assembly._row_exceptions(row)
    row.candidate_errors = dict(reversed(list(row.candidate_errors.items())))
    assert assembly._row_exceptions(row) == first
    assert [item["reason"] for item in first] == [
        "Candidate 1: Invalid",
        "Candidate 2: Malformed alternate",
        "Candidate 10: Malformed alternate",
    ]


@pytest.fixture
def stage_outputs(settings: Settings) -> dict[str, Any]:
    cases = [
        ("z", 10, "resolved", "same-id"),
        ("a", 10, "weak", "same-id"),
        ("a", 2, "resolved", "same-id"),
        ("z", 2, "no_match", None),
        ("a", 3, "error", None),
    ]
    inputs = [
        InputEntity(
            sheet=sheet,
            row_number=number,
            name=f"Input {sheet}{number}",
            address="原始地址",
            country="SGP",
        )
        for sheet, number, _, _ in cases
    ]
    resolved = [
        ResolvedEntity(
            sheet=source.sheet,
            row_number=source.row_number,
            input_name=source.name,
            status=status,
            entity_id=entity_id,
        )
        for source, (_, _, status, entity_id) in zip(inputs, cases, strict=True)
    ]
    resolved[0].candidate_errors = {2: "Malformed alternate"}
    resolved[2].profile_error = "Profile unavailable"
    resolved[4].error = "Resolution failed"
    resolved[4].error_type = "SyntheticResolutionError"
    ingested = {
        sheet: IngestResult(
            entities=[row for row in inputs if row.sheet == sheet],
            exceptions=[{"sheet": sheet, "row_number": "20", "kind": kind, "reason": reason}],
        )
        for sheet, kind, reason in [
            ("z", "dropped", "Missing name"),
            ("a", "annotated", "Invalid country"),
        ]
    }
    return {
        "generated_at": "injected-artifact-time",
        "ingested": ingested,
        "inputs": inputs,
        "resolved": resolved,
        "profiles": {
            "same-id": EntityProfile(
                entity_id="same-id",
                label="供应商",
                translated_label="Supplier",
                countries=["SGP"],
                psa_count=2,
                degree=8,
                risk_factors=[],
                max_level=None,
            )
        },
        "upstream": {
            "same-id": UpstreamResult(
                supplier_id="same-id",
                entities={},
                partial_results=False,
                explored_count=None,
                status="error",
            )
        },
        "psa_records": [
            {
                "portfolio": "z",
                "row_number": 10,
                "entity_id": "same-id",
                "psa_count": 2,
                "psa_risky": False,
                "psa_status": "available",
            },
            {
                "portfolio": "a",
                "row_number": 2,
                "entity_id": "same-id",
                "psa_count": None,
                "psa_risky": None,
                "psa_status": "unavailable",
            },
        ],
        "ranked": [{"portfolio": "z", "upstream_id": "canonical-node", "suppliers": []}],
        "convergence": {"z": {"funnel": {"after_severity_filter": 1}}},
        "coverage": {"z": {"error": 1, "not_attempted": 1}},
        "headline_portfolio": "z",
        "limit": 5,
        "settings": settings,
        "ontology": synthetic_ontology(),
    }


def test_assembly_is_deterministic_and_does_not_mutate_inputs(
    stage_outputs: dict[str, Any],
) -> None:
    # Assembling twice leaves the stage inputs unchanged and returns identical evidence.
    before = deepcopy(stage_outputs)
    first = assemble_findings(**stage_outputs)
    second = assemble_findings(**stage_outputs)
    assert stage_outputs == before
    assert first == second
    assert first.generated_at == "injected-artifact-time"
    for field, parameter in [
        ("shared_nodes", "ranked"),
        ("convergence", "convergence"),
        ("coverage", "coverage"),
    ]:
        assert getattr(first, field) == stage_outputs[parameter]
    assert cast(dict[str, Any], first.manifest)["sheets"] == ["z", "a"]
    assert cast(dict[str, Any], first.manifest)["headline_portfolio"] == "z"
    assert (
        cast(dict[str, Any], first.manifest)["input_entities"]
        == cast(dict[str, Any], first.manifest)["limit"]
        == 5
    )


def test_supplier_row_accounting_numeric_sort_and_exact_ordered_schema(
    stage_outputs: dict[str, Any],
) -> None:
    # Every input row survives assembly, with a stable set of keys and in numeric row order.
    result = assemble_findings(**stage_outputs)
    assert len(cast(list[dict[str, Any]], result.suppliers)) == len(stage_outputs["inputs"])
    assert [
        (r["portfolio"], r["row_number"]) for r in cast(list[dict[str, Any]], result.suppliers)
    ] == [
        ("a", 2),
        ("a", 3),
        ("a", 10),
        ("z", 2),
        ("z", 10),
    ]
    keys = [
        "row_number",
        "sheet",
        "input_name",
        "entity_id",
        "label",
        "translated_label",
        "match_strength",
        "score",
        "candidate_count",
        "candidates",
        "candidate_errors",
        "status",
        "error",
        "error_type",
        "profile_error",
        "profile_error_type",
        "portfolio",
        "input_address",
        "input_country",
        "resolution_status",
        "profile_status",
        "profile",
        "coverage_status",
        "upstream_entity_count",
        "upstream_partial_results",
        "upstream_error_type",
        "psa_count",
        "psa_risky",
        "psa_status",
    ]
    # Frozen post-R1 evidence has 29 keys, including the three PSA fields.
    assert len(keys) == 29
    assert all(list(row) == keys for row in cast(list[dict[str, Any]], result.suppliers))
    available = cast(list[dict[str, Any]], result.suppliers)[-1]
    unavailable, _, weak, unmatched = cast(list[dict[str, Any]], result.suppliers)[:4]
    assert available["profile"]["label"] == "供应商"
    assert available["input_address"] == "原始地址"
    assert available["psa_count"] == 2 and available["psa_risky"] is False
    assert unavailable["psa_status"] == "unavailable"
    assert unavailable["status"] == unavailable["profile_status"] == "error"
    assert unavailable["resolution_status"] == "resolved"
    assert unavailable["profile"] is unavailable["psa_count"] is unavailable["psa_risky"] is None
    for row in (weak, unmatched):
        assert row["profile"] is row["coverage_status"] is row["upstream_entity_count"] is None
        assert row["psa_status"] == "not_resolved"


@pytest.mark.parametrize(
    "change, message",
    [("identity", "PSA entity identity differs"), ("leftover", "PSA evidence has no matching")],
)
def test_psa_invariants(stage_outputs: dict[str, Any], change: str, message: str) -> None:
    # PSA evidence for the wrong entity or row is rejected, and the stage inputs are left unchanged.
    record = stage_outputs["psa_records"][0]
    record["entity_id" if change == "identity" else "row_number"] = (
        "wrong-id" if change == "identity" else 99
    )
    before = deepcopy(stage_outputs)
    with pytest.raises(ValueError, match=message):
        assemble_findings(**stage_outputs)
    assert stage_outputs == before


def test_exception_shapes_and_order_are_ingestion_then_input_row(
    stage_outputs: dict[str, Any],
) -> None:
    # Each stage's diagnostics keep their own shape and always come out in the same order.
    result = assemble_findings(**stage_outputs)
    ingestion = [
        {
            "sheet": "z",
            "row_number": "20",
            "kind": "dropped",
            "reason": "Missing name",
            "stage": "ingestion",
        },
        {
            "sheet": "a",
            "row_number": "20",
            "kind": "annotated",
            "reason": "Invalid country",
            "stage": "ingestion",
        },
    ]
    expected = [
        (
            "z",
            "10",
            "upstream",
            "Upstream retrieval failed; bounded evidence unavailable",
            "UpstreamError",
        ),
        ("z", "10", "resolution", "Candidate 2: Malformed alternate", "ValidationError"),
        ("a", "10", "resolution", "Weak match: needs adjudication; profile not requested", "weak"),
        (
            "a",
            "2",
            "upstream",
            "Upstream retrieval failed; bounded evidence unavailable",
            "UpstreamError",
        ),
        ("a", "2", "profile", "Profile unavailable", "ProfileError"),
        ("z", "2", "resolution", "No matching entity; profile not requested", "no_match"),
        ("a", "3", "resolution", "Resolution failed", "SyntheticResolutionError"),
    ]
    assert result.exceptions == ingestion + [
        {
            "sheet": sheet,
            "row_number": number,
            "input_name": f"Input {sheet}{number}",
            "stage": stage,
            "reason": reason,
            "error_type": error,
        }
        for sheet, number, stage, reason, error in expected
    ]


def test_mismatched_stage_row_counts_raise(stage_outputs: dict[str, Any]) -> None:
    # Stages with different row counts raise an error instead of silently truncating assembly.
    stage_outputs["resolved"].pop()
    with pytest.raises(ValueError, match="zip"):
        assemble_findings(**stage_outputs)


def test_assembly_imports_only_leaf_modules_and_typing_support() -> None:
    # Findings assembly does not depend on the retrieval or presentation layers.
    tree = ast.parse(Path(assembly.__file__).read_text(encoding="utf-8"))
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
    assert imports <= {
        "collections.abc",
        "json",
        "typing",
        "sayari_poc.models",
        "sayari_poc.config",
        "sayari_poc.risk_taxonomy",
    }

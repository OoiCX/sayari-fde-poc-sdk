"""The supplier worklist, checked against evidence small enough to count by hand.

Asserting the rendered table against the rollup that produced it would be tautological, so every
expectation here is derived from the fixture. The cases that silently corrupt a worklist each get a
test: a row whose identity was never accepted, a row measured on partial evidence, and two workbook
rows resolving to the same entity.
"""

from typing import Any

import pytest

from sayari_poc.models import (
    EntityProfile,
    OntologyFactor,
    ResolvedEntity,
    UpstreamEntity,
    UpstreamResult,
)
from sayari_poc.risk_taxonomy import RiskOntology
from sayari_poc.rollups import NOT_ATTEMPTED, supplier_breakdown


def factor(slug: str, level: str) -> OntologyFactor:
    return OntologyFactor.model_validate(
        {
            "id": slug,
            "label": "Published " + slug,
            "description": "Synthetic definition for " + slug + ".",
            "categories": ["synthetic"],
            "level": level,
            "risk_type": "network",
        }
    )


@pytest.fixture
def ontology() -> RiskOntology:
    factors = {
        "crit": factor("crit", "critical"),
        "high": factor("high", "high"),
        "mild": factor("mild", "elevated"),
    }
    return RiskOntology(factors, {"snapshot": "synthetic", "factor_count": len(factors)})


def row(number: int, entity_id: str | None, status: str = "resolved") -> ResolvedEntity:
    return ResolvedEntity.model_validate(
        {
            "row_number": number,
            "sheet": "list_3",
            "input_name": f"Input {number}",
            "entity_id": entity_id,
            "label": f"Supplier {entity_id}" if entity_id else None,
            "status": status,
        }
    )


def profile(entity_id: str, slugs: dict[str, str]) -> EntityProfile:
    return EntityProfile.model_validate(
        {
            "entity_id": entity_id,
            "label": entity_id,
            "translated_label": None,
            "countries": ["USA"],
            "psa_count": 0,
            "degree": 0,
            "risk_factors": [
                {"factor": slug, "value": True, "level": level, "metadata": {}}
                for slug, level in slugs.items()
            ],
            "max_level": None,
        }
    )


def retrieval(entity_id: str, status: str, entities: int) -> UpstreamResult:
    return UpstreamResult.model_validate(
        {
            "supplier_id": entity_id,
            "entities": {
                f"u{index}": UpstreamEntity.model_validate(
                    {
                        "entity_id": f"u{index}",
                        "label": f"Upstream {index}",
                        "translated_label": None,
                        "countries": ["CHN"],
                        "country_count": 1,
                        "risk_factors": [],
                    }
                )
                for index in range(entities)
            },
            "partial_results": status == "partial",
            "explored_count": entities,
            "status": status,
        }
    )


def node(upstream_id: str, supplier_ids: list[str]) -> dict[str, Any]:
    return {
        "upstream_id": upstream_id,
        "portfolio": "list_3",
        "suppliers": [{"entity_id": entity_id} for entity_id in supplier_ids],
    }


def by_label(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(record["label"]): record for record in rows}


def test_a_row_whose_identity_was_never_accepted_is_still_listed(ontology: RiskOntology) -> None:
    """An unresolved row must appear as unmeasured, never be dropped from the worklist."""
    rows = supplier_breakdown("list_3", [row(1, None, "no_match")], {}, {}, [], ontology)
    assert len(rows) == 1
    # It keeps the workbook name, because there is no resolved label to show.
    assert rows[0]["label"] == "Input 1"
    # Its state is distinct from a retrieval that ran and returned nothing.
    assert rows[0]["coverage"] == NOT_ATTEMPTED
    assert rows[0]["upstream"] is None and rows[0]["flagged"] is None


def test_partial_evidence_is_carried_beside_the_counts(ontology: RiskOntology) -> None:
    """A count on partial evidence is a floor; the row must say so."""
    rows = supplier_breakdown(
        "list_3",
        [row(1, "e1")],
        {"e1": profile("e1", {"high": "high"})},
        {"e1": retrieval("e1", "partial", 3)},
        [node("u1", ["e1"])],
        ontology,
    )
    assert rows[0]["coverage"] == "partial"
    assert rows[0]["upstream"] == 3
    assert rows[0]["flagged"] == 1


def test_two_workbook_rows_sharing_one_entity_both_appear(ontology: RiskOntology) -> None:
    """Duplicate canonical IDs cannot collapse distinct workbook input rows."""
    rows = supplier_breakdown(
        "list_3",
        [row(1, "e1"), row(2, "e1")],
        {"e1": profile("e1", {"high": "high"})},
        {"e1": retrieval("e1", "assessed", 2)},
        [node("u1", ["e1"])],
        ontology,
    )
    assert len(rows) == 2
    # Both carry the same evidence, because they are the same canonical identity.
    assert {int(str(record["flagged"])) for record in rows} == {1}


def test_flagged_counts_how_many_shared_entities_a_supplier_reaches(
    ontology: RiskOntology,
) -> None:
    # A supplier's flagged total is the number of shared nodes it connects to.
    rows = by_label(
        supplier_breakdown(
            "list_3",
            [row(1, "e1"), row(2, "e2")],
            {},
            {entity_id: retrieval(entity_id, "assessed", 2) for entity_id in ("e1", "e2")},
            [node("u1", ["e1", "e2"]), node("u2", ["e1"])],
            ontology,
        )
    )
    # e1 reaches both flagged entities; e2 reaches only the shared one.
    assert rows["Supplier e1"]["flagged"] == 2
    assert rows["Supplier e2"]["flagged"] == 1


def test_severity_is_counted_through_the_ontology_not_the_profile_level(
    ontology: RiskOntology,
) -> None:
    """A row must never read as severe here while the selection rule excludes it."""
    # The profile claims critical; the ontology publishes the factor as elevated.
    rows = supplier_breakdown(
        "list_3",
        [row(1, "e1")],
        {"e1": profile("e1", {"mild": "critical"})},
        {},
        [],
        ontology,
    )
    assert rows[0]["severe_factors"] == 0
    assert rows[0]["critical_factors"] == 0


def test_critical_is_carried_separately_so_it_cannot_be_buried(ontology: RiskOntology) -> None:
    """The column combines critical and high; the marker keeps a critical visible."""
    rows = supplier_breakdown(
        "list_3",
        [row(1, "e1")],
        {"e1": profile("e1", {"crit": "critical", "high": "high", "mild": "elevated"})},
        {},
        [],
        ontology,
    )
    # Two of the three published factors are at a selected level.
    assert rows[0]["severe_factors"] == 2
    # And the report can still say one of them is critical.
    assert rows[0]["critical_factors"] == 1
    # The elevated factor is counted separately: it supports, it does not qualify.
    assert rows[0]["elevated_factors"] == 1


def test_elevated_factors_keep_a_row_from_reading_as_clean(ontology: RiskOntology) -> None:
    """Zero selected factors plus several elevated ones is not a clean row."""
    rows = supplier_breakdown(
        "list_3",
        [row(1, "e1")],
        {"e1": profile("e1", {"mild": "elevated"})},
        {},
        [],
        ontology,
    )
    # Nothing qualifies this supplier for review...
    assert rows[0]["severe_factors"] == 0
    # ...but the row still reports the evidence Sayari returned.
    assert rows[0]["elevated_factors"] == 1


def test_only_the_requested_supplier_list_is_reported(ontology: RiskOntology) -> None:
    # The worklist leaves out input rows that belong to other portfolios.
    other = ResolvedEntity.model_validate(
        {
            "row_number": 1,
            "sheet": "list_1",
            "input_name": "Elsewhere",
            "entity_id": "e9",
            "status": "resolved",
        }
    )
    rows = supplier_breakdown("list_3", [row(1, "e1"), other], {}, {}, [], ontology)
    assert [record["entity_id"] for record in rows] == ["e1"]


def test_bars_scale_against_the_busiest_row(ontology: RiskOntology) -> None:
    # Bar widths are scaled to the busiest row and rounded the same way every time.
    rows = by_label(
        supplier_breakdown(
            "list_3",
            [row(1, "e1"), row(2, "e2"), row(3, "e3")],
            {},
            {entity_id: retrieval(entity_id, "assessed", 3) for entity_id in ("e1", "e2", "e3")},
            [node("u1", ["e1", "e2"]), node("u2", ["e1"]), node("u3", ["e1", "e2"])],
            ontology,
        )
    )
    # The busiest row fills the track; the others are a share of it, not of any total.
    assert rows["Supplier e1"]["flagged"] == 3 and rows["Supplier e1"]["flagged_share"] == 100
    assert rows["Supplier e2"]["flagged"] == 2 and rows["Supplier e2"]["flagged_share"] == 67
    assert rows["Supplier e3"]["flagged"] == 0 and rows["Supplier e3"]["flagged_share"] == 0


def test_ordering_is_total_and_stable(ontology: RiskOntology) -> None:
    """Byte-gated artifacts require a total order, so ties must break on label and id."""
    rows = [row(1, "e1"), row(2, "e2")]
    order = [
        str(record["label"]) for record in supplier_breakdown("list_3", rows, {}, {}, [], ontology)
    ]
    # Both rows tie on every count, so the label decides and repeats identically.
    assert order == ["Supplier e1", "Supplier e2"]
    reversed_rows = supplier_breakdown("list_3", list(reversed(rows)), {}, {}, [], ontology)
    assert order == [str(record["label"]) for record in reversed_rows]


def test_an_empty_supplier_list_does_not_divide_by_zero(ontology: RiskOntology) -> None:
    # An empty worklist returns cleanly instead of computing a scale from nothing.
    assert supplier_breakdown("list_3", [], {}, {}, [], ontology) == []


@pytest.mark.parametrize("status", ["weak", "no_match", "error"])
def test_unaccepted_row_cannot_borrow_a_resolved_rows_evidence(
    ontology: RiskOntology, status: str
) -> None:
    # Sharing a candidate ID with a resolved row gives an unaccepted row none of its evidence.
    rows = supplier_breakdown(
        "list_3",
        [row(1, "e1"), row(2, "e1", status)],
        {"e1": profile("e1", {"high": "high"})},
        {"e1": retrieval("e1", "assessed", 3)},
        [node("u1", ["e1"])],
        ontology,
    )
    assert rows[0]["coverage"] == "assessed" and rows[0]["flagged"] == 1
    unaccepted = rows[1]
    assert unaccepted["label"] == "Input 2"
    assert unaccepted["entity_id"] is None
    assert unaccepted["coverage"] == NOT_ATTEMPTED
    for key in ("upstream", "flagged", "severe_factors", "critical_factors", "elevated_factors"):
        assert unaccepted[key] is None


def test_failed_upstream_counts_are_unknown_even_when_a_profile_is_available(
    ontology: RiskOntology,
) -> None:
    # Having a profile does not turn the counts from a failed upstream retrieval into zeros.
    record = supplier_breakdown(
        "list_3",
        [row(1, "e1")],
        {"e1": profile("e1", {})},
        {"e1": retrieval("e1", "error", 0)},
        [],
        ontology,
    )[0]
    assert record["coverage"] == "error"
    assert record["upstream"] is None and record["flagged"] is None
    assert record["severe_factors"] == record["critical_factors"] == record["elevated_factors"] == 0
    assert record["flagged_share"] == 0


def test_successful_empty_retrieval_and_profile_are_measured_zero(ontology: RiskOntology) -> None:
    # Successful but empty responses give measured zeros, not unknowns.
    record = supplier_breakdown(
        "list_3",
        [row(1, "e1")],
        {"e1": profile("e1", {})},
        {"e1": retrieval("e1", "no_data", 0)},
        [],
        ontology,
    )[0]
    assert record["coverage"] == "no_data"
    for key in ("upstream", "flagged", "severe_factors", "critical_factors", "elevated_factors"):
        assert record[key] == 0


def test_failed_profile_cannot_borrow_another_rows_factor_counts(ontology: RiskOntology) -> None:
    # If a row's profile failed, its factor counts stay unknown even when another row shares its ID.
    accepted, failed = row(1, "e1"), row(2, "e1")
    failed.profile_error = "Profile unavailable"
    records = supplier_breakdown(
        "list_3",
        [accepted, failed],
        {"e1": profile("e1", {"crit": "critical"})},
        {"e1": retrieval("e1", "assessed", 2)},
        [],
        ontology,
    )
    assert records[0]["critical_factors"] == 1
    assert records[1]["coverage"] == "assessed" and records[1]["upstream"] == 2
    for key in ("severe_factors", "critical_factors", "elevated_factors"):
        assert records[1][key] is None


def test_worklist_counts_other_entities_without_discarding_the_traversal_root(
    ontology: RiskOntology,
) -> None:
    # The upstream count leaves out the supplier itself, but the raw evidence still keeps it.
    result = retrieval("e1", "assessed", 2)
    result.entities["e1"] = result.entities["u0"].model_copy(update={"entity_id": "e1"})
    record = supplier_breakdown("list_3", [row(1, "e1")], {}, {"e1": result}, [], ontology)[0]
    assert record["upstream"] == 2
    assert len(result.entities) == 3 and "e1" in result.entities

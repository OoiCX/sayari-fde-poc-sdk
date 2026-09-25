"""Traversal anchors and profile evidence cannot fabricate upstream convergence."""

from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from sayari_poc.analysis import (
    build_warehouse,
    classify_risk_factors,
    convergence_funnel,
    coverage_summary,
    shared_nodes,
    suppressed_hubs,
)
from sayari_poc.config import Settings
from sayari_poc.findings import assemble_findings
from sayari_poc.identity import psa_exposure
from sayari_poc.models import IngestResult, InputEntity, RiskFactor
from sayari_poc.pipeline import _convergence
from tests.ontology_support import synthetic_ontology
from tests.unit.test_p2_stages import client_for, upstream_payload
from tests.unit.test_schema_load import node, profile, result, supplier


def test_traversal_root_response_retains_entity_without_self_edge(tmp_path: Path) -> None:
    # The traversal root is stored as an entity, but never as an edge to itself.
    payload = upstream_payload("S1")
    payload["data"]["entities"]["S1"] = {
        **payload["data"]["entities"]["node"],
        "id": "S1",
    }
    client, _ = client_for(tmp_path, payload)
    with client:
        retrieval = client.upstream("S1")
    assert retrieval.status == "assessed"
    with closing(
        build_warehouse(tmp_path / "root.duckdb", [supplier()], {}, {"S1": retrieval})
    ) as con:
        assert con.execute("SELECT upstream_id FROM upstream_entities ORDER BY 1").fetchall() == [
            ("S1",),
            ("node",),
        ]
        assert con.execute("SELECT * FROM supplier_upstream ORDER BY 1, 2").fetchall() == [
            ("S1", "node")
        ]


@pytest.mark.parametrize("other_count", [0, 1, 2])
def test_shared_requires_two_other_suppliers(tmp_path: Path, other_count: int) -> None:
    # A node needs two other suppliers to count as shared; being its own root does not count.
    ids = [f"S{i + 1}" for i in range(other_count + 1)]
    rows = [supplier(entity_id, i + 2) for i, entity_id in enumerate(ids)]
    upstream = {entity_id: result(entity_id, node("S1")) for entity_id in ids}
    with closing(build_warehouse(tmp_path / "shared.duckdb", rows, {}, upstream)) as con:
        classify_risk_factors(con, synthetic_ontology())
        selected = shared_nodes(con, 10, "list_3")
        assert selected["supplier_count"].tolist() == ([2] if other_count == 2 else [])
        assert convergence_funnel(con, 10, "list_3")["shared_nodes"] == int(other_count == 2)


def test_generated_findings_never_list_node_as_its_own_supplier(tmp_path: Path) -> None:
    # Findings never list a node as its own supplier or count it in its own upstream total.
    rows = [supplier(f"S{i}", i + 1) for i in range(1, 4)]
    upstream = {f"S{i}": result(f"S{i}", node("S1")) for i in range(1, 4)}
    ontology = synthetic_ontology()
    with closing(build_warehouse(tmp_path / "findings.duckdb", rows, {}, upstream)) as con:
        classify_risk_factors(con, ontology)
        ranked, convergence = _convergence(con, ["list_3"], 10)
        coverage = coverage_summary(con)
    inputs = [
        InputEntity(
            sheet=row.sheet,
            row_number=row.row_number,
            name=row.input_name,
            address=None,
            country=None,
        )
        for row in rows
    ]
    findings = assemble_findings(
        generated_at="synthetic",
        ingested={"list_3": IngestResult(entities=inputs, exceptions=[])},
        inputs=inputs,
        resolved=rows,
        profiles={},
        upstream=upstream,
        psa_records=psa_exposure(rows, {}, ontology).to_dict("records"),
        ranked=ranked,
        convergence=convergence,
        coverage=coverage,
        headline_portfolio="list_3",
        limit=None,
        settings=Settings(_env_file=None),
        ontology=ontology,
    )
    assert len(findings.shared_nodes) == 1
    assert [row["upstream_entity_count"] for row in findings.suppliers] == [0, 1, 1]
    for shared in findings.shared_nodes:
        assert shared["upstream_id"] not in [
            s["entity_id"] for s in cast(list[dict[str, object]], shared["suppliers"])
        ]


@pytest.mark.parametrize("upstream_factors", [[], ["uflpa", "ordinary_factor"]])
def test_shared_factor_selection_ignores_profile_only_rows(
    tmp_path: Path, upstream_factors: list[str]
) -> None:
    # Only factors reported by the traversal can qualify a shared upstream node.
    entity = node("S1").model_copy(update={"risk_factors": upstream_factors})
    p = profile().model_copy(
        update={
            "risk_factors": [
                RiskFactor(factor=slug, level="high", metadata={})
                for slug in ["owned_by_soe", "exports_ilab_child_labor"]
            ]
        }
    )
    with closing(
        build_warehouse(
            tmp_path / "provenance.duckdb",
            [supplier(f"S{i}", i + 1) for i in range(1, 4)],
            {"S1": p},
            {f"S{i}": result(f"S{i}", entity) for i in (2, 3)},
        )
    ) as con:
        classify_risk_factors(con, synthetic_ontology())
        assert len(shared_nodes(con, 10, "list_3")) == int(bool(upstream_factors))
        assert convergence_funnel(con, 10, "list_3")["after_severity_filter"] == int(
            bool(upstream_factors)
        )
        evidence = shared_nodes(con, 10, "list_3").to_dict("records")
        if upstream_factors:
            assert evidence[0]["severe_factors"] == ["uflpa"]
            assert evidence[0]["other_factors"] == ["ordinary_factor"]
        else:
            assert evidence == []
        hub = suppressed_hubs(con, 1, "list_3", 10).to_dict("records")[0]
        assert hub["severe_factors"] == (["uflpa"] if upstream_factors else [])
        assert hub["other_factors"] == (["ordinary_factor"] if upstream_factors else [])
        assert con.execute(
            "SELECT factor, level FROM risk_factors WHERE source='profile' ORDER BY factor"
        ).fetchall() == [("exports_ilab_child_labor", "high"), ("owned_by_soe", "high")]


def test_profile_collision_keeps_upstream_qualification_and_raw_profile_level(
    tmp_path: Path,
) -> None:
    # If a profile repeats an upstream factor, the node still qualifies and keeps the profile level.
    entity = node("S1").model_copy(update={"risk_factors": ["uflpa"]})
    p = profile().model_copy(
        update={"risk_factors": [RiskFactor(factor="uflpa", level="relevant", metadata={})]}
    )
    with closing(
        build_warehouse(
            tmp_path / "overlap.duckdb",
            [supplier(f"S{i}", i + 1) for i in range(1, 4)],
            {"S1": p},
            {f"S{i}": result(f"S{i}", entity) for i in (2, 3)},
        )
    ) as con:
        assert con.execute("SELECT source, level FROM risk_factors").fetchone() == (
            "upstream",
            "relevant",
        )
        classify_risk_factors(con, synthetic_ontology())
        assert shared_nodes(con, 10, "list_3")["supplier_count"].tolist() == [2]
        assert shared_nodes(con, 10, "list_3").iloc[0]["severe_factors"] == ["uflpa"]


def test_r18_model_comment_keeps_original_without_redundant_sentence() -> None:
    # The relevance comment in models.py keeps the rule and drops the redundant sentence.
    source = Path("src/sayari_poc/models.py").read_text(encoding="utf-8")
    assert "# Relevance is separate from the categorical match strength;" in source
    assert "# Retain the finite relevance value without rescaling;" not in source


def test_r18_autoescape_comment_explains_template_suffix_security() -> None:
    # The autoescape comment in report.py explains why the .j2 suffix needs an explicit policy.
    source = Path("src/sayari_poc/report.py").read_text(encoding="utf-8")
    assert "# The HTML and SVG template filenames end in .j2, so extension-based" in source
    assert "# them. Turn escaping on for every packaged template." in source


def test_r18_retry_comment_keeps_original_without_duplicate_citation() -> None:
    # The retry comment in transport.py says the SDK owns retry scheduling, not the audit telemetry.
    source = Path("src/sayari_poc/transport.py").read_text(encoding="utf-8")
    assert "We only label attempts we observe; the SDK still decides whether to retry" in " ".join(
        line.strip().lstrip("# ") for line in source.splitlines()
    )
    assert (
        "# Verified in sayari/core/http_client.py: _should_retry and HttpClient.request own retry"
        not in source
    )

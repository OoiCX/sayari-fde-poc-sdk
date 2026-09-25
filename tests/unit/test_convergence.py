"""Synthetic convergence cases run against the actual Task 8 schema, without API access."""

import json
from collections.abc import Iterator
from contextlib import closing
from inspect import signature
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import duckdb
import pytest

from sayari_poc.analysis import (
    UnclassifiedRiskFactors,
    build_warehouse,
    classify_risk_factors,
    create_schema,
    sensitivity_sweep,
    shared_nodes,
)
from sayari_poc.models import (
    EntityProfile,
    Findings,
    ResolvedEntity,
    RiskFactor,
    UpstreamEntity,
    UpstreamResult,
)
from sayari_poc.risk_taxonomy import RiskOntology
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import AuditedTransport
from tests.ontology_support import synthetic_ontology

COLUMNS = [
    "upstream_id",
    "supplier_count",
    "label",
    "translated_label",
    "countries",
    "country_count",
    "portfolio",
    "suppliers",
    "severe_factors",
    "other_factors",
]


@pytest.fixture(autouse=True)
def forbid_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    guard = Mock(side_effect=AssertionError("Convergence must never access the API"))
    monkeypatch.setattr(SayariClient, "__init__", guard)
    monkeypatch.setattr(AuditedTransport, "handle_request", guard)
    yield
    guard.assert_not_called()


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    with closing(duckdb.connect(":memory:")) as connection:
        create_schema(connection)
        for row, entity_id in enumerate(["S1", "S2", "S3"], start=1):
            add_supplier(connection, entity_id, row)
        yield connection


def add_supplier(
    con: duckdb.DuckDBPyConnection,
    entity_id: str,
    row: int,
    portfolio: str = "list_3",
    status: str = "resolved",
) -> None:
    con.execute(
        "INSERT INTO suppliers (portfolio, row_number, entity_id, input_name, resolution_status) "
        "VALUES (?, ?, ?, ?, ?)",
        [portfolio, row, entity_id, entity_id, status],
    )


def add_node(
    con: duckdb.DuckDBPyConnection,
    node_id: str = "U1",
    supplier_ids: tuple[str, ...] = ("S1", "S2"),
    country_count: int = 1,
    factors: tuple[str, ...] = ("owned_by_military_civil_fusion",),
) -> None:
    con.execute(
        "INSERT INTO upstream_entities VALUES (?, ?, ?, ?, ?)",
        [node_id, "\u77f3\u5bb6\u5e84", "Shijiazhuang", ["CHN"] * country_count, country_count],
    )
    for supplier_id in supplier_ids:
        con.execute("INSERT INTO supplier_upstream VALUES (?, ?)", [supplier_id, node_id])
    for factor in factors:
        con.execute(
            "INSERT INTO risk_factors (entity_id, factor, source) VALUES (?, ?, 'upstream')",
            [node_id, factor],
        )


@pytest.fixture(autouse=True)
def source_ontology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sayari_poc.analysis.load_ontology", synthetic_ontology)


@pytest.mark.parametrize(
    "level,expected", [("critical", True), ("high", True), ("elevated", False), ("relevant", False)]
)
@pytest.mark.parametrize("kind", ["seed", "network", "psa"])
def test_classification_uses_published_fields(
    con: duckdb.DuckDBPyConnection, level: str, kind: str, expected: bool
) -> None:
    # Eligibility comes from the published level and risk type, never from how a factor is spelled.
    from sayari_poc.models import OntologyFactor

    factor = OntologyFactor.model_validate(
        {
            "id": "ordinary",
            "label": "Published",
            "description": "Definition",
            "categories": [],
            "level": level,
            "risk_type": kind,
        }
    )
    add_node(con, factors=("ordinary",))
    ontology = RiskOntology({"ordinary": factor}, {})
    assert classify_risk_factors(con, ontology) == 1
    assert con.execute(
        "SELECT is_severe, ontology_level, risk_type, ontology_status FROM risk_factors"
    ).fetchall() == [(expected, level, kind, "resolved")]
    assert len(shared_nodes(con, 10, "list_3")) == int(expected)


@pytest.mark.parametrize("supplier_ids,expected", [(("S1",), 0), (("S1", "S2"), 1)])
def test_shared_means_two_distinct_suppliers(
    con: duckdb.DuckDBPyConnection, supplier_ids: tuple[str, ...], expected: int
) -> None:
    # A node counts as shared only when it links to two or more distinct suppliers.
    add_node(con, supplier_ids=supplier_ids)
    classify_risk_factors(con)
    assert len(shared_nodes(con, 10, portfolio="list_3")) == expected


@pytest.mark.parametrize("countries,expected", [(0, 1), (9, 1), (10, 1), (11, 0), (59, 0)])
def test_hub_suppression_boundary(
    con: duckdb.DuckDBPyConnection, countries: int, expected: int
) -> None:
    # The country limit is inclusive: a node at exactly the limit stays, only larger counts drop.
    add_node(con, country_count=countries)
    classify_risk_factors(con)
    assert len(shared_nodes(con, 10, portfolio="list_3")) == expected


@pytest.mark.parametrize(
    "factors,expected",
    [
        (("owned_by_military_civil_fusion",), 1),
        (("exports_ilab_forced_labor",), 0),
        (("unknown_factor",), 0),
        ((), 0),
        (("uflpa", "exports_ilab_forced_labor"), 1),
        (("xinjiang_ilab_forced_labor",), 0),
        (("export_to_soe", "owned_by_soe"), 1),
    ],
)
def test_severity_is_an_exists_filter_from_ontology(
    con: duckdb.DuckDBPyConnection, factors: tuple[str, ...], expected: int
) -> None:
    # A selected factor qualifies a node, and having several does not multiply its supplier count.
    add_node(con, factors=factors)
    con.execute("UPDATE upstream_entities SET label='Xinjiang military', translated_label='uflpa'")
    con.execute("UPDATE risk_factors SET level='critical'")
    classify_risk_factors(con)
    result = shared_nodes(con, 10, "list_3")
    assert len(result) == expected
    if expected:
        assert result.iloc[0]["supplier_count"] == 2


def test_convergence_never_crosses_portfolios(con: duckdb.DuckDBPyConnection) -> None:
    # A supplier in another portfolio can never make a node count as shared.
    con.execute("UPDATE suppliers SET portfolio='another_portfolio' WHERE entity_id='S2'")
    add_node(con)
    classify_risk_factors(con)
    for portfolio in ["another_portfolio", "list_3"]:
        assert shared_nodes(con, 10, portfolio=portfolio).empty


def test_arbitrary_portfolios_are_independent_and_parameters_are_bound(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Each portfolio sees only its own evidence, and a SQL-like portfolio name is treated as data.
    for row, entity_id in enumerate(["C1", "C2", "S1"], start=1):
        add_supplier(con, entity_id, row, "another_portfolio")
    add_node(con, "BOTH", ("S1", "S2", "S3", "C1"))
    add_node(con, "SECONDARY", ("C1", "C2"))
    classify_risk_factors(con)
    assert shared_nodes(con, 10, "list_3")[["upstream_id", "supplier_count"]].values.tolist() == [
        ["BOTH", 3]
    ]
    assert shared_nodes(con, 10, "another_portfolio")[
        ["upstream_id", "supplier_count"]
    ].values.tolist() == [
        ["BOTH", 2],
        ["SECONDARY", 2],
    ]
    assert shared_nodes(con, 10, "list_3' OR true --").empty
    assert shared_nodes(con, 10, "absent").empty


@pytest.mark.parametrize("status", ["weak", "no_match", "error"])
def test_unresolved_candidate_cannot_contribute_to_supplier_count(
    con: duckdb.DuckDBPyConnection, status: str
) -> None:
    # A candidate that was not accepted cannot borrow a resolved supplier's membership.
    add_node(con)
    add_supplier(con, "S1", 1, "another_portfolio", status)
    add_supplier(con, "S2", 2, "another_portfolio", status)
    classify_risk_factors(con)
    assert shared_nodes(con, 10, "another_portfolio").empty
    assert shared_nodes(con, 10, "list_3").iloc[0]["supplier_count"] == 2
    con.execute(
        "UPDATE suppliers SET resolution_status='resolved' WHERE portfolio='another_portfolio' "
        "AND entity_id='S1'"
    )
    assert shared_nodes(con, 10, "another_portfolio").empty


def test_duplicate_input_rows_and_edges_cannot_inflate_count(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Repeated input rows and edges never create extra canonical suppliers.
    add_supplier(con, "S1", 4)
    add_node(con, supplier_ids=("S1",))
    classify_risk_factors(con)
    assert shared_nodes(con, 10, "list_3").empty
    # The real schema rejects duplicate edges; DISTINCT handles fan-out from repeated input rows.
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO supplier_upstream VALUES ('S1', 'U1')")
    con.execute("INSERT INTO supplier_upstream VALUES ('S2', 'U1')")
    assert shared_nodes(con, 10, "list_3").iloc[0]["supplier_count"] == 2


def test_ranking_uses_supplier_count_then_countries_then_canonical_id(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Ranking breaks ties explicitly, so reviewers see the same order on every run.
    add_node(con, "Z", ("S1", "S2", "S3"), 10)
    add_node(con, "B", country_count=1)
    add_node(con, "A", country_count=1)
    add_node(con, "C", country_count=2)
    classify_risk_factors(con)
    result = shared_nodes(con, 10, "list_3")
    assert result["upstream_id"].tolist() == ["Z", "A", "B", "C"]
    assert result["supplier_count"].tolist() == [3, 2, 2, 2]
    assert result["country_count"].tolist() == [10, 1, 1, 2]
    assert list(result.columns) == COLUMNS  # No derived score or coverage column.


def test_any_unclassified_row_raises_even_outside_the_requested_portfolio(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Any unclassified row blocks queries, even in another portfolio, so no evidence is hidden.
    con.execute(
        "INSERT INTO risk_factors (entity_id, factor, source) "
        "VALUES ('UNRELATED', 'unknown', 'profile')"
    )
    with pytest.raises(UnclassifiedRiskFactors, match="classify_risk_factors"):
        shared_nodes(con, 10, "absent")
    assert classify_risk_factors(con) == 1
    assert shared_nodes(con, 10, "absent").empty


def test_classification_populates_all_rows_preserves_evidence_and_reapplies(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Reclassifying updates every occurrence of a factor and leaves the raw evidence unchanged.
    add_node(con, factors=("uflpa", "exports_ilab_child_labor", "unknown_factor"))
    con.execute(
        "INSERT INTO risk_factors (entity_id, factor, source, level) "
        "VALUES ('S1', 'uflpa', 'profile', 'high')"
    )
    raw_query = "SELECT entity_id, factor, source, level FROM risk_factors ORDER BY 1, 2"
    raw = con.execute(raw_query).fetchall()
    assert classify_risk_factors(con) == 4
    classified = con.execute("SELECT * FROM risk_factors ORDER BY 1, 2").fetchall()
    assert [(row[1], row[4], row[7]) for row in classified] == [
        ("uflpa", True, "resolved"),
        ("exports_ilab_child_labor", False, "resolved"),
        ("uflpa", True, "resolved"),
        ("unknown_factor", None, "unresolved"),
    ]
    assert classify_risk_factors(con) == 4
    assert con.execute("SELECT * FROM risk_factors ORDER BY 1, 2").fetchall() == classified
    con.execute("UPDATE risk_factors SET is_severe=false, ontology_level='relevant'")
    assert classify_risk_factors(con) == 4
    assert con.execute("SELECT * FROM risk_factors ORDER BY 1, 2").fetchall() == classified
    assert con.execute(raw_query).fetchall() == raw
    assert con.execute(
        "SELECT COUNT(*) FROM risk_factors WHERE ontology_status IS NULL"
    ).fetchone() == (0,)
    con.execute(
        "INSERT INTO risk_factors (entity_id, factor, source) "
        "VALUES ('NEW', 'unknown_factor', 'upstream')"
    )
    with pytest.raises(UnclassifiedRiskFactors):
        shared_nodes(con, 10, "list_3")
    assert classify_risk_factors(con) == 5
    assert len(shared_nodes(con, 10, "list_3")) == 1


def test_empty_warehouse_and_required_portfolio(con: duckdb.DuckDBPyConnection) -> None:
    # An empty warehouse still returns the declared columns, and a portfolio is always required.
    assert classify_risk_factors(con) == 0
    result = shared_nodes(con, 10, "list_3")
    assert result.empty
    assert list(result.columns) == COLUMNS
    assert signature(shared_nodes).parameters["portfolio"].default is signature(shared_nodes).empty
    with pytest.raises(TypeError):
        cast(Any, shared_nodes)(con, 10)


def test_loaded_bounded_evidence_persists_classification_without_changing_coverage(
    tmp_path: Path,
) -> None:
    # Classifying the warehouse does not change the coverage state recorded at retrieval.
    path = tmp_path / "convergence.duckdb"
    suppliers = [
        ResolvedEntity(
            row_number=i,
            sheet="list_3",
            input_name=entity_id,
            entity_id=entity_id,
            status="resolved",
        )
        for i, entity_id in enumerate(["S1", "S2", "S1", "EMPTY", "FAILED", "UNATTEMPTED"], 1)
    ]
    node = UpstreamEntity(
        entity_id="U",
        label="\u77f3\u5bb6\u5e84",
        translated_label=None,
        countries=["CHN"],
        country_count=1,
        risk_factors=["owned_by_soe", "owned_by_soe", "unknown"],
    )
    upstream = {
        entity_id: UpstreamResult(
            supplier_id=entity_id,
            entities={"U": node},
            partial_results=True,
            explored_count=1000000,
            status="partial",
        )
        for entity_id in ["S1", "S2"]
    }
    upstream["EMPTY"] = UpstreamResult(
        supplier_id="EMPTY", entities={}, partial_results=False, explored_count=0, status="no_data"
    )
    upstream["FAILED"] = UpstreamResult(
        supplier_id="FAILED",
        entities={},
        partial_results=False,
        explored_count=None,
        status="error",
        error_type="Timeout",
    )
    profile = EntityProfile(
        entity_id="S1",
        label=None,
        translated_label=None,
        countries=[],
        degree=None,
        psa_count=0,
        max_level="high",
        risk_factors=[RiskFactor(factor="esg_score", level="high", metadata={})],
    )
    with closing(build_warehouse(path, suppliers, {"S1": profile}, upstream)) as con:
        raw_tables = {
            table: con.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
            for table in ["suppliers", "upstream_entities", "supplier_upstream"]
        }
        with pytest.raises(UnclassifiedRiskFactors):
            shared_nodes(con, 10, "list_3")
        assert classify_risk_factors(con) == 3
        result = shared_nodes(con, 10, "list_3")
        assert list(result.columns) == COLUMNS
        assert result.iloc[0]["upstream_id"] == "U"
        assert result.iloc[0]["supplier_count"] == 2
        assert result.iloc[0]["label"] == "\u77f3\u5bb6\u5e84"
        assert result.iloc[0]["translated_label"] is None
        assert list(result.iloc[0]["countries"]) == ["CHN"]
        for table, rows in raw_tables.items():
            assert con.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall() == rows
    with closing(duckdb.connect(str(path))) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM risk_factors WHERE ontology_status IS NULL"
        ).fetchone() == (0,)
        assert shared_nodes(con, 10, "list_3")["supplier_count"].tolist() == [2]


@pytest.mark.parametrize("countries", [[], ["CHN", "CHN", "MYS"], None])
def test_shared_evidence_serializes_without_losing_countries(
    con: duckdb.DuckDBPyConnection, countries: list[str] | None
) -> None:
    # Serializing a shared node keeps its full country list and its supplier count.
    add_node(con)
    con.execute(
        "UPDATE upstream_entities SET countries = ?, country_count = ?",
        [countries, len(countries) if countries is not None else 0],
    )
    classify_risk_factors(con)
    result = shared_nodes(con, 10, "list_3")
    assert list(result.columns) == COLUMNS
    records = result.to_dict(orient="records")
    restored = json.loads(json.dumps(records, ensure_ascii=False))
    assert restored[0]["countries"] == countries
    assert restored[0]["supplier_count"] == 2
    assert restored[0]["label"] == "\u77f3\u5bb6\u5e84"
    assert restored[0]["translated_label"] == "Shijiazhuang"
    findings = Findings(generated_at="synthetic", suppliers=[], shared_nodes=records)
    assert json.loads(findings.model_dump_json())["shared_nodes"] == restored
    assert json.loads(json.dumps(shared_nodes(con, 10, "absent").to_dict(orient="records"))) == []


@pytest.mark.parametrize("previously_classified", [False, True])
def test_failed_classification_cannot_persist_a_partial_taxonomy(
    con: duckdb.DuckDBPyConnection, previously_classified: bool
) -> None:
    # A failed classification update leaves the previous taxonomy intact.
    con.execute(
        "INSERT INTO risk_factors (entity_id, factor, source) VALUES "
        "('FIRST', 'a_uflpa', 'upstream'), ('SECOND', 'b_uflpa', 'upstream'), "
        "('SECOND', 'z_uflpa', 'upstream')"
    )
    if previously_classified:
        con.execute("UPDATE risk_factors SET is_severe=(factor='z_uflpa'), ontology_level='high'")
    before = con.execute("SELECT * FROM risk_factors ORDER BY factor").fetchall()
    # Make the update fail on SECOND after FIRST could already have been written. This test-only
    # constraint is deliberately absent from the production schema.
    con.execute("CREATE UNIQUE INDEX injected_failure ON risk_factors(entity_id, is_severe)")
    with pytest.raises(duckdb.ConstraintException):
        classify_risk_factors(con)
    assert con.execute("SELECT * FROM risk_factors ORDER BY factor").fetchall() == before
    con.execute("DROP INDEX injected_failure")
    assert classify_risk_factors(con) == 3
    assert con.execute(
        "SELECT DISTINCT is_severe, ontology_status FROM risk_factors"
    ).fetchall() == [(True, "resolved")]


def test_classification_respects_caller_transaction(con: duckdb.DuckDBPyConnection) -> None:
    # When the caller rolls back its transaction, the classification changes roll back too.
    add_node(con)
    con.begin()
    con.execute(
        "INSERT INTO risk_factors (entity_id, factor, source) VALUES ('EXTRA', 'uflpa', 'upstream')"
    )
    assert classify_risk_factors(con) == 2
    assert len(shared_nodes(con, 10, "list_3")) == 1
    con.rollback()
    assert con.execute("SELECT is_severe, ontology_status FROM risk_factors").fetchall() == [
        (None, None)
    ]
    with pytest.raises(UnclassifiedRiskFactors):
        shared_nodes(con, 10, "list_3")


def test_soe_trade_evidence_is_retained_when_not_severe(con: duckdb.DuckDBPyConnection) -> None:
    # Trade factors that are not selected stay as evidence but never enter the review queue.
    add_node(con, factors=("export_to_soe", "exports_ilab_forced_labor"))
    classify_risk_factors(con)
    assert con.execute(
        "SELECT factor, is_severe, ontology_status FROM risk_factors ORDER BY factor"
    ).fetchall() == [
        ("export_to_soe", False, "resolved"),
        ("exports_ilab_forced_labor", False, "resolved"),
    ]
    assert con.execute("SELECT COUNT(*) FROM upstream_entities").fetchone() == (1,)
    assert con.execute("SELECT COUNT(*) FROM supplier_upstream").fetchone() == (2,)
    assert shared_nodes(con, 10, portfolio="list_3").empty


def test_sensitivity_sweep_counts_retained_and_excluded_against_hand_built_evidence(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Pin the sweep to evidence whose answer is countable by hand.

    The sweep is retained in findings.json for audit, so asserting it against its own output would
    be tautological - an off-by-one would pass unnoticed. These expectations are derived from the
    fixture, not from the implementation.
    """
    # Four qualifying entities, one per country band, plus one that carries no selected factor.
    for index, countries in enumerate((3, 8, 12, 18), start=1):
        add_node(con, node_id=f"Q{index}", country_count=countries)
    add_node(con, node_id="N1", country_count=3, factors=("unlisted_factor",))
    classify_risk_factors(con)

    sweep = sensitivity_sweep(con, "list_3", (5, 10, 15, 20))

    # Retained counts qualifying entities at or below each limit: {3}, {3,8}, {3,8,12}, all four.
    # Excluded is the rest of the four-entity qualifying population, never the unlisted one.
    assert sweep == {
        "5": {"retained": 1, "excluded_qualifying": 3},
        "10": {"retained": 2, "excluded_qualifying": 2},
        "15": {"retained": 3, "excluded_qualifying": 1},
        "20": {"retained": 4, "excluded_qualifying": 0},
    }
    # The qualifying population is a property of the evidence, not of any single threshold.
    assert {row["retained"] + row["excluded_qualifying"] for row in sweep.values()} == {4}

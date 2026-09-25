"""Coverage counts input rows and preserves bounded upstream outcomes independently."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest

from sayari_poc.analysis import build_warehouse, coverage_summary, create_schema
from sayari_poc.models import Findings, ResolvedEntity, UpstreamEntity, UpstreamResult
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import AuditedTransport

STATUS_KEYS = ["assessed", "partial", "no_data", "error", "not_attempted"]


@pytest.fixture(autouse=True)
def forbid_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    guard = Mock(side_effect=AssertionError("Coverage must never access the API"))
    monkeypatch.setattr(SayariClient, "__init__", guard)
    monkeypatch.setattr(AuditedTransport, "handle_request", guard)
    yield
    guard.assert_not_called()


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    with closing(duckdb.connect(":memory:")) as connection:
        create_schema(connection)
        yield connection


def insert_supplier(
    con: duckdb.DuckDBPyConnection,
    row: int,
    status: str | None,
    portfolio: str = "list_3",
    entity_id: str | None = "S1",
    resolution: str = "resolved",
) -> None:
    con.execute(
        "INSERT INTO suppliers (portfolio, row_number, entity_id, input_name, label, "
        "resolution_status, coverage_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [portfolio, row, entity_id, "\u77f3\u5bb6\u5e84", "M\u00fcnchen", resolution, status],
    )


@pytest.fixture
def coverage_db(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    for row, status in enumerate(["assessed", "partial", "no_data", "error"], 1):
        insert_supplier(con, row, status, entity_id=f"S{row}")
    insert_supplier(con, 5, None, entity_id="candidate", resolution="weak")
    insert_supplier(con, 6, None, entity_id=None, resolution="no_match")
    insert_supplier(con, 7, "assessed")  # A second input row shares S1's retrieval.
    insert_supplier(con, 1, "assessed", portfolio="another_portfolio")
    insert_supplier(con, 2, "error", portfolio="another_portfolio", entity_id="C2")
    con.execute(
        "INSERT INTO upstream_entities VALUES "
        "('U1',NULL,NULL,[],0), ('U2',NULL,NULL,[],0), ('U3',NULL,NULL,[],0)"
    )
    con.execute("INSERT INTO supplier_upstream VALUES ('S1','U1'),('S1','U2'),('S2','U3')")
    con.execute(
        "INSERT INTO risk_factors (entity_id,factor,source) VALUES "
        "('U1','uflpa','upstream'),('U1','unknown','upstream'),"
        "('U2','esg_score','upstream')"
    )
    return con


def test_all_outcomes_and_input_row_denominator_are_portfolio_scoped(
    coverage_db: duckdb.DuckDBPyConnection,
) -> None:
    # In each portfolio, the five coverage states add up to that portfolio's workbook rows.
    summary = coverage_summary(coverage_db)
    assert summary == {
        "another_portfolio": {
            "assessed": 1,
            "partial": 0,
            "no_data": 0,
            "error": 1,
            "not_attempted": 0,
        },
        "list_3": {"assessed": 2, "partial": 1, "no_data": 1, "error": 1, "not_attempted": 2},
    }
    totals = dict(
        coverage_db.execute(
            "SELECT portfolio, COUNT(*) FROM suppliers GROUP BY portfolio"
        ).fetchall()
    )
    assert set(summary) == set(totals)
    for portfolio, counts in summary.items():
        assert sum(counts.values()) == totals[portfolio]
        assert list(counts) == STATUS_KEYS
        assert all(type(count) is int for count in counts.values())


@pytest.mark.parametrize("status", ["assessed", "partial", "no_data", "error", None])
def test_each_status_counts_only_its_own_category_with_explicit_zeros(
    con: duckdb.DuckDBPyConnection, status: str | None
) -> None:
    # Each outcome lands in exactly one bucket, and every other bucket is reported as an explicit 0.
    insert_supplier(con, 1, status)
    expected = dict.fromkeys(STATUS_KEYS, 0)
    expected[status if status is not None else "not_attempted"] = 1
    assert coverage_summary(con) == {"list_3": expected}


def test_edges_and_risks_cannot_multiply_counts_or_be_required(
    coverage_db: duckdb.DuckDBPyConnection,
) -> None:
    # Coverage counts input rows, so extra edges or risk factors never change it.
    before = coverage_summary(coverage_db)
    # Repeated evidence can neither inflate counts nor change coverage categories.
    coverage_db.execute("INSERT INTO supplier_upstream VALUES ('S1','U1') ON CONFLICT DO NOTHING")
    coverage_db.execute("INSERT INTO supplier_upstream VALUES ('S1','U3')")
    coverage_db.execute(
        "INSERT INTO risk_factors (entity_id,factor,source) "
        "VALUES ('S1','another_factor','profile')"
    )
    assert coverage_summary(coverage_db) == before
    for table in ["supplier_upstream", "upstream_entities", "risk_factors"]:
        coverage_db.execute(f"DROP TABLE {table}")
    assert coverage_summary(coverage_db) == before


@pytest.mark.parametrize("status", ["unexpected", "not_attempted", "", "ASSESSED", " partial "])
def test_invalid_persisted_status_raises_without_normalizing_or_dropping_it(
    con: duckdb.DuckDBPyConnection, status: str
) -> None:
    # An unknown stored status raises an error instead of quietly dropping out of coverage.
    insert_supplier(con, 1, "assessed")
    insert_supplier(con, 1, status, portfolio="another_portfolio")
    with pytest.raises(ValueError) as error:
        coverage_summary(con)
    assert repr(status) in str(error.value)


def test_empty_warehouse_has_no_synthetic_portfolios(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # An empty warehouse reports no portfolios rather than inventing zero totals.
    assert coverage_summary(con) == {}


def test_summary_preserves_source_rows_and_has_stable_json_order(
    coverage_db: duckdb.DuckDBPyConnection,
) -> None:
    # Summarizing coverage leaves the source rows untouched, and the JSON order never varies.
    rows = coverage_db.execute("SELECT * FROM suppliers ORDER BY portfolio,row_number").fetchall()
    summary = coverage_summary(coverage_db)
    assert list(summary) == ["another_portfolio", "list_3"]
    encoded = Findings(generated_at="synthetic", suppliers=[], coverage=summary).model_dump_json()
    assert (
        coverage_db.execute("SELECT * FROM suppliers ORDER BY portfolio,row_number").fetchall()
        == rows
    )
    coverage_db.execute("DELETE FROM suppliers")
    coverage_db.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows[::-1])
    coverage_db.execute(
        "UPDATE suppliers SET input_name='uflpa', label=NULL, translated_label='Different label'"
    )
    reordered = coverage_summary(coverage_db)
    assert Findings(
        generated_at="synthetic", suppliers=[], coverage=reordered
    ).model_dump_json() == (encoded)


def test_loaded_earlier_failures_remain_not_attempted(
    tmp_path: Path,
) -> None:
    # Rows that failed earlier count as not_attempted, never as a successful empty (no_data).
    rows = [
        ResolvedEntity(
            row_number=1,
            sheet="list_3",
            input_name="Weak",
            entity_id="S1",
            status="weak",
            match_strength="weak",
        ),
        ResolvedEntity(row_number=2, sheet="list_3", input_name="No match", status="no_match"),
        ResolvedEntity(
            row_number=3,
            sheet="list_3",
            input_name="Resolution failure",
            status="error",
            error_type="Timeout",
            error="Synthetic resolution failure",
        ),
        ResolvedEntity(
            row_number=4,
            sheet="list_3",
            input_name="Profile failure",
            status="resolved",
            entity_id="PROFILE",
            profile_error="Synthetic profile failure",
            profile_error_type="Timeout",
        ),
        ResolvedEntity(
            row_number=5,
            sheet="list_3",
            input_name="Skipped",
            status="resolved",
            entity_id="SKIPPED",
        ),
        ResolvedEntity(
            row_number=6,
            sheet="another_portfolio",
            input_name="Retrieved",
            status="resolved",
            entity_id="S1",
        ),
    ]
    node = UpstreamEntity(entity_id="U1", countries=[], country_count=0, risk_factors=[])
    upstream = {
        "S1": UpstreamResult(
            supplier_id="S1",
            entities={"U1": node},
            status="assessed",
            partial_results=False,
            explored_count=42,
        )
    }
    originals = [row.model_copy(deep=True) for row in rows]
    with closing(build_warehouse(tmp_path / "coverage.duckdb", rows, {}, upstream)) as con:
        before = con.execute("SELECT * FROM suppliers ORDER BY portfolio,row_number").fetchall()
        assert coverage_summary(con) == {
            "another_portfolio": {
                "assessed": 1,
                "partial": 0,
                "no_data": 0,
                "error": 0,
                "not_attempted": 0,
            },
            "list_3": {"assessed": 0, "partial": 0, "no_data": 0, "error": 0, "not_attempted": 5},
        }
        assert con.execute("SELECT * FROM suppliers ORDER BY portfolio,row_number").fetchall() == (
            before
        )
    assert rows == originals

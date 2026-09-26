"""Synthetic warehouse contracts; every owned connection is closed for Windows reruns."""

from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest

from sayari_poc.analysis import build_warehouse, create_schema
from sayari_poc.models import (
    EntityProfile,
    ResolvedEntity,
    UpstreamEntity,
    UpstreamResult,
)
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import AuditedTransport
from tests.warehouse_support import node, profile, result, supplier

SampleInputs = tuple[list[ResolvedEntity], dict[str, EntityProfile], dict[str, UpstreamResult]]

TABLES = {"suppliers", "upstream_entities", "supplier_upstream", "risk_factors", "supply_paths"}


@pytest.fixture(autouse=True)
def forbid_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    guard = Mock(side_effect=AssertionError("Warehouse must never access the API"))
    monkeypatch.setattr(SayariClient, "__init__", guard)
    monkeypatch.setattr(AuditedTransport, "handle_request", guard)
    yield
    guard.assert_not_called()


@pytest.fixture
def sample_inputs() -> SampleInputs:
    rows = [supplier(f"S{i}", i + 1) for i in range(1, 7)]
    rows.extend(
        [
            supplier("S1", 8),
            supplier("S1", 2, "another_portfolio"),
            supplier("candidate", 9).model_copy(
                update={"status": "weak", "match_strength": "weak"}
            ),
            ResolvedEntity(row_number=10, sheet="list_3", input_name="Unknown", status="no_match"),
            ResolvedEntity(row_number=11, sheet="list_3", input_name="Failed", status="error"),
        ]
    )
    upstream = {f"S{i}": result(f"S{i}", node()) for i in range(1, 4)}
    upstream["S2"] = upstream["S2"].model_copy(
        update={"status": "partial", "partial_results": True}
    )
    upstream["S4"] = result("S4")
    upstream["S5"] = result("S5").model_copy(
        update={"status": "error", "error_type": "SayariError"}
    )
    return rows, {"S1": profile()}, upstream


def snapshot(con: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[object, ...]]]:
    return {
        "suppliers": con.execute(
            "SELECT * FROM suppliers ORDER BY portfolio, row_number"
        ).fetchall(),
        "upstream_entities": con.execute(
            "SELECT * FROM upstream_entities ORDER BY upstream_id"
        ).fetchall(),
        "supplier_upstream": con.execute(
            "SELECT * FROM supplier_upstream ORDER BY 1, 2"
        ).fetchall(),
        "risk_factors": con.execute("SELECT * FROM risk_factors ORDER BY 1, 2").fetchall(),
        "supply_paths": con.execute(
            "SELECT * FROM supply_paths ORDER BY source_entity_id, path_index, hop_position"
        ).fetchall(),
    }


def test_real_schema_columns_and_null_defaults() -> None:
    # The warehouse schema has exactly the declared columns, and unknown values default to null.
    with closing(duckdb.connect(":memory:")) as con:
        create_schema(con)
        assert {row[0] for row in con.execute("SHOW TABLES").fetchall()} == TABLES
        columns = con.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "ORDER BY ordinal_position"
        ).fetchall()
        assert [name for table, name in columns if table == "supplier_upstream"] == [
            "supplier_id",
            "upstream_id",
        ]
        assert {table for table, name in columns if name == "portfolio"} == {"suppliers"}
        assert "degree" not in {name for _, name in columns}
        assert {name for table, name in columns if table == "suppliers"} == {
            "portfolio",
            "row_number",
            "entity_id",
            "input_name",
            "label",
            "translated_label",
            "match_strength",
            "resolution_status",
            "coverage_status",
        }
        assert {name for table, name in columns if table == "upstream_entities"} == {
            "upstream_id",
            "label",
            "translated_label",
            "countries",
            "country_count",
        }
        assert {name for table, name in columns if table == "risk_factors"} == {
            "entity_id",
            "factor",
            "source",
            "level",
            "is_severe",
            "ontology_level",
            "risk_type",
            "ontology_status",
        }
        assert con.execute(
            "SELECT is_nullable, column_default, data_type FROM information_schema.columns "
            "WHERE table_name='risk_factors' AND column_name='is_severe'"
        ).fetchall() == [("YES", None, "BOOLEAN")]


def test_suppliers_coverage_shared_nodes_and_raw_risks(
    tmp_path: Path, sample_inputs: SampleInputs
) -> None:
    # Loading keeps the input identities, the coverage states and the raw risk evidence.
    with closing(build_warehouse(tmp_path / "warehouse.duckdb", *sample_inputs)) as con:
        data = snapshot(con)
        assert len(data["suppliers"]) == len(sample_inputs[0]) == 11
        assert con.execute(
            "SELECT entity_id, coverage_status FROM suppliers "
            "WHERE portfolio='list_3' AND row_number < 8 ORDER BY row_number"
        ).fetchall() == [
            ("S1", "assessed"),
            ("S2", "partial"),
            ("S3", "assessed"),
            ("S4", "no_data"),
            ("S5", "error"),
            ("S6", None),
        ]
        assert con.execute(
            "SELECT resolution_status, coverage_status FROM suppliers "
            "WHERE row_number >= 9 ORDER BY row_number"
        ).fetchall() == [("weak", None), ("no_match", None), ("error", None)]
        assert con.execute(
            "SELECT label, translated_label, input_name, match_strength FROM suppliers "
            "WHERE portfolio='list_3' AND row_number=2"
        ).fetchone() == ("Profile S1", "Translated S1", "Supplier S1", "strong")
        assert data["upstream_entities"] == [
            ("U1", node().label, "Shijiazhuang Taimington", ["CHN", "MYS"], 2)
        ]
        assert data["supplier_upstream"] == [("S1", "U1"), ("S2", "U1"), ("S3", "U1")]
        assert data["risk_factors"] == [
            ("S1", "raw_profile_factor", "profile", "high", None, None, None, None),
            ("U1", "owned_by_military_civil_fusion", "upstream", None, None, None, None, None),
            ("U1", "unknown_factor", "upstream", None, None, None, None, None),
        ]


def test_missing_upstream_labels_and_countries(tmp_path: Path) -> None:
    # Missing labels and an empty country list load into the warehouse unchanged.
    bare = UpstreamEntity(entity_id="U", countries=[], country_count=0, risk_factors=[])
    with closing(
        build_warehouse(tmp_path / "bare.duckdb", [supplier()], {}, {"S1": result("S1", bare)})
    ) as con:
        assert con.execute("SELECT * FROM upstream_entities").fetchone() == ("U", None, None, [], 0)


def test_duplicate_inputs_edges_and_factors_are_idempotent(
    tmp_path: Path, sample_inputs: SampleInputs
) -> None:
    # Repeating identical evidence never adds rows to any warehouse table.
    rows, profiles, upstream = sample_inputs
    rows.append(rows[0].model_copy(deep=True))
    profiles["S1"].risk_factors.append(profiles["S1"].risk_factors[0].model_copy(deep=True))
    for retrieval in upstream.values():
        for entity in retrieval.entities.values():
            entity.risk_factors.append(entity.risk_factors[0])
    path = tmp_path / "reload.duckdb"
    with closing(build_warehouse(path, *sample_inputs)) as con:
        first = snapshot(con)
        assert [
            len(first[table])
            for table in [
                "suppliers",
                "upstream_entities",
                "supplier_upstream",
                "risk_factors",
                "supply_paths",
            ]
        ] == [11, 1, 3, 3, 0]
        con.execute("ALTER TABLE upstream_entities ADD COLUMN obsolete INTEGER")
        con.execute("UPDATE risk_factors SET is_severe=false, ontology_level='relevant'")
    with closing(build_warehouse(path, *sample_inputs)) as con:
        assert snapshot(con) == first
    # Reversing the input order must not change the stored evidence.
    with closing(
        build_warehouse(
            path,
            list(reversed(rows)),
            profiles,
            dict(reversed(list(upstream.items()))),
        )
    ) as con:
        assert snapshot(con) == first


def test_profile_level_keeps_upstream_source_and_unclassified_state(tmp_path: Path) -> None:
    # Profile enrichment keeps the upstream source on shared factors and leaves them unclassified.
    p = profile()
    n = node("S1")
    n.risk_factors = ["raw_profile_factor", "unknown_factor"]
    with closing(
        build_warehouse(
            tmp_path / "collision.duckdb",
            [supplier(), supplier("S2", 3)],
            {"S1": p},
            {"S2": result("S2", n)},
        )
    ) as con:
        assert con.execute("SELECT * FROM risk_factors ORDER BY factor").fetchall() == [
            ("S1", "raw_profile_factor", "upstream", "high", None, None, None, None),
            ("S1", "unknown_factor", "upstream", None, None, None, None, None),
        ]


def test_weak_row_does_not_inherit_retrieval_for_same_resolved_id(tmp_path: Path) -> None:
    # A weak row stays not attempted, even though its candidate ID matches a resolved row.
    weak = supplier(row=3).model_copy(update={"status": "weak", "match_strength": "weak"})
    with closing(
        build_warehouse(
            tmp_path / "weak.duckdb", [supplier(), weak], {}, {"S1": result("S1", node())}
        )
    ) as con:
        assert con.execute(
            "SELECT coverage_status FROM suppliers ORDER BY row_number"
        ).fetchall() == [("assessed",), (None,)]


@pytest.mark.parametrize(
    "conflict",
    [
        "supplier_row",
        "upstream_label",
        "upstream_label_case",
        "upstream_countries",
        "upstream_factors",
        "profile_factor",
        "profile_key",
        "upstream_key",
        "node_key",
        "country_count",
        "orphan_profile",
        "orphan_upstream",
        "error_with_nodes",
        "no_data_with_nodes",
        "blank_supplier_id",
    ],
)
def test_conflicting_or_inconsistent_inputs_fail_without_destroying_existing_build(
    tmp_path: Path, sample_inputs: SampleInputs, conflict: str
) -> None:
    # Conflicting evidence makes the load roll back and leaves the previous warehouse intact.
    path = tmp_path / "atomic.duckdb"
    with closing(build_warehouse(path, *sample_inputs)) as con:
        before = snapshot(con)
    rows, profiles, upstream = sample_inputs
    if conflict == "supplier_row":
        rows.append(rows[0].model_copy(update={"input_name": "Conflicting name"}))
    elif conflict == "upstream_label":
        upstream["S2"].entities["U1"].label = "Conflicting label"
    elif conflict == "upstream_label_case":
        upstream["S1"].entities["U1"].label = "Native Label"
        upstream["S2"].entities["U1"].label = "native label"
    elif conflict == "upstream_countries":
        upstream["S2"].entities["U1"].countries = ["CHN", "USA"]
    elif conflict == "upstream_factors":
        upstream["S2"].entities["U1"].risk_factors = ["different_evidence"]
    elif conflict == "profile_factor":
        profiles["S1"].risk_factors.append(
            profiles["S1"].risk_factors[0].model_copy(update={"level": "critical"})
        )
    elif conflict == "profile_key":
        profiles["S1"].entity_id = "other"
    elif conflict == "upstream_key":
        upstream["S1"].supplier_id = "other"
    elif conflict == "node_key":
        upstream["S1"].entities["U1"].entity_id = "other"
    elif conflict == "country_count":
        upstream["S1"].entities["U1"].country_count = 99
    elif conflict == "orphan_profile":
        profiles["other"] = profile("other")
    elif conflict == "orphan_upstream":
        upstream["other"] = result("other", node())
    elif conflict in {"error_with_nodes", "no_data_with_nodes"}:
        upstream["S1"].status = "error" if conflict == "error_with_nodes" else "no_data"
    else:
        rows[0].entity_id = " "
    with pytest.raises(ValueError):
        build_warehouse(path, *sample_inputs)
    with closing(duckdb.connect(str(path))) as con:
        assert snapshot(con) == before
    # The failed loader must not leave a connection holding the file open.
    path.unlink()


def test_empty_rebuild_removes_old_rows(tmp_path: Path, sample_inputs: SampleInputs) -> None:
    # Rebuilding with empty inputs removes every stale row from the earlier build.
    path = tmp_path / "empty.duckdb"
    with closing(build_warehouse(path, *sample_inputs)):
        pass
    with closing(build_warehouse(path, [], {}, {})) as con:
        assert all(not rows for rows in snapshot(con).values())


def test_caller_path_and_parameterized_source_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The warehouse is written where the caller asks, and SQL-like source text is stored as data.
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "chosen"
    output.mkdir()
    path = output / "custom.duckdb"
    quoted = supplier("S'; DROP TABLE suppliers; --")
    quoted.input_name = "O'Brien \u77f3\u5bb6\u5e84"
    with closing(build_warehouse(path, [quoted], {}, {})) as con:
        assert con.execute("SELECT entity_id, input_name FROM suppliers").fetchone() == (
            quoted.entity_id,
            quoted.input_name,
        )
        assert {row[0] for row in con.execute("SHOW TABLES").fetchall()} == TABLES
    assert list(tmp_path.rglob("*.duckdb")) == [path]
    path.unlink()


@pytest.mark.parametrize("reordered", ["countries", "risk_factors", "both"])
def test_shared_node_list_order_is_canonical_and_inputs_are_unchanged(
    tmp_path: Path, reordered: str
) -> None:
    # Lists are stored in canonical order, and the source objects are left unchanged.
    first = node()
    first.countries.reverse()
    first.risk_factors.reverse()
    second = first.model_copy(deep=True)
    if reordered in {"countries", "both"}:
        second.countries.reverse()
    if reordered in {"risk_factors", "both"}:
        second.risk_factors.reverse()
    before = [first.model_copy(deep=True), second.model_copy(deep=True)]
    suppliers = [supplier(), supplier("S2", 3)]
    path = tmp_path / "reordered.duckdb"
    with closing(
        build_warehouse(
            path, suppliers, {}, {"S1": result("S1", first), "S2": result("S2", second)}
        )
    ) as con:
        loaded = snapshot(con)
        assert loaded["upstream_entities"] == [
            ("U1", first.label, first.translated_label, ["CHN", "MYS"], 2)
        ]
        assert loaded["supplier_upstream"] == [("S1", "U1"), ("S2", "U1")]
        assert loaded["risk_factors"] == [
            ("U1", "owned_by_military_civil_fusion", "upstream", None, None, None, None, None),
            ("U1", "unknown_factor", "upstream", None, None, None, None, None),
        ]
    # Swap which supplier provides each ordering, not just the dictionary insertion order.
    with closing(
        build_warehouse(
            path,
            list(reversed(suppliers)),
            {},
            {"S2": result("S2", first), "S1": result("S1", second)},
        )
    ) as con:
        assert snapshot(con) == loaded
    assert [first, second] == before


def test_country_sorting_preserves_source_multiplicity_and_count(tmp_path: Path) -> None:
    # Sorting countries keeps the duplicates and the country count as received.
    entity = node()
    entity.countries = ["MYS", "CHN", "MYS"]
    entity.country_count = 3
    with closing(
        build_warehouse(
            tmp_path / "countries.duckdb", [supplier()], {}, {"S1": result("S1", entity)}
        )
    ) as con:
        assert con.execute("SELECT countries, country_count FROM upstream_entities").fetchone() == (
            ["CHN", "MYS", "MYS"],
            3,
        )
    assert entity.countries == ["MYS", "CHN", "MYS"]


def test_relationship_join_isolates_portfolios_with_a_shared_node(tmp_path: Path) -> None:
    # A canonical node shared across portfolios never merges their supplier sets.
    rows = [
        supplier(),
        supplier("S2", 3),
        supplier("SECONDARY", 2, "another_portfolio"),
        supplier("S1", 4),
    ]
    upstream = {entity_id: result(entity_id, node()) for entity_id in ["S1", "S2", "SECONDARY"]}
    with closing(build_warehouse(tmp_path / "portfolios.duckdb", rows, {}, upstream)) as con:
        for portfolio, expected in [
            ("list_3", [("S1", "U1"), ("S2", "U1")]),
            ("another_portfolio", [("SECONDARY", "U1")]),
            ("absent", []),
        ]:
            assert (
                con.execute(
                    "SELECT DISTINCT e.supplier_id, e.upstream_id FROM supplier_upstream e "
                    "JOIN suppliers s ON s.entity_id = e.supplier_id "
                    "WHERE s.portfolio = ? ORDER BY e.supplier_id",
                    [portfolio],
                ).fetchall()
                == expected
            )
        assert con.execute("SELECT COUNT(*) FROM upstream_entities").fetchone() == (1,)
        assert con.execute("SELECT COUNT(*) FROM supplier_upstream").fetchone() == (3,)

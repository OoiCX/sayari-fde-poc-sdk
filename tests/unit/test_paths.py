"""R4a source fidelity, supplier-preserving selection and offline artifact contracts."""

import json
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import duckdb
import httpx
import pytest

from sayari_poc import pipeline
from sayari_poc.analysis import build_warehouse, create_schema
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.findings import PATH_MAX_PER_NODE, assemble_findings
from sayari_poc.models import (
    Findings,
    PathComponent,
    SupplyPath,
    SupplyPathHop,
    UpstreamResult,
)
from sayari_poc.report import report_datetime
from sayari_poc.sayari_sdk import SayariClient
from tests.ontology_support import synthetic_ontology
from tests.pipeline_support import _node, _upstream
from tests.sdk_support import QueuedCache
from tests.warehouse_support import supplier

COMPONENT_KEYS = ["hs_code", "departure_countries", "arrival_countries", "min_date", "max_date"]
MANIFEST_KEYS = [
    "sheets",
    "limit",
    "input_entities",
    "headline_portfolio",
    "hub_max_countries",
    "path_max_per_node",
    "max_upstream_depth",
    "upstream_limit",
    "taxonomy",
]
OLD_FINDINGS_KEYS = [
    "generated_at",
    "suppliers",
    "shared_nodes",
    "convergence",
    "coverage",
    "exceptions",
    "manifest",
]
_MISSING = object()


def component(hs_code: str | None = "8542") -> dict[str, Any]:
    return {
        "hs_code": hs_code,
        "departure_countries": ["USA", "CHN", "USA"],
        "arrival_countries": ["MEX", "USA"],
        "min_date": "2023-01-05",
        "max_date": "2025-06-17",
    }


def record(source: str = "s1", ids: tuple[str, ...] = ("u2", "u3", "node")) -> dict[str, Any]:
    return {
        "source_entity_id": source,
        "path": [
            {
                "tier": tier,
                "entity_id": entity_id,
                "components": [component("8708"), component("8542"), component("8708")],
            }
            for tier, entity_id in zip((2, 3, 4), ids, strict=False)
        ],
    }


def data_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {hop["entity_id"] for path in records for hop in path["path"]}
    return {
        "entities": {entity_id: _node(entity_id, 1, []) for entity_id in sorted(ids)},
        "paths": records,
    }


def fetch_result(
    data: dict[str, Any], source: str = "s1", *, partial: bool = False, explored: int = 0
) -> UpstreamResult:
    """Exercise the real SDK with raw synthetic bytes; no production cache is opened."""
    settings = Settings(_env_file=None, sayari_client_id=None, sayari_client_secret=None)
    cache = QueuedCache(
        Path("unused"),
        ({"filters": {}, "data": data, "partial_results": partial, "explored_count": explored},),
    )
    with SayariClient(settings, cache, offline=True) as client:
        result = client.upstream(source)
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 0
        return result


def retrieval(source: str, records: list[dict[str, Any]]) -> UpstreamResult:
    result = fetch_result(data_for(records), source)
    assert result.error_type is None
    return result


def ranked_node(entity_id: str, sources: list[str], portfolio: str = "list_3") -> dict[str, Any]:
    return {
        "portfolio": portfolio,
        "upstream_id": entity_id,
        "suppliers": [{"entity_id": source} for source in sources],
    }


def assemble(
    settings: Settings,
    upstream: dict[str, UpstreamResult],
    ranked: list[dict[str, Any]],
    headline: str | None = "list_3",
) -> Findings:
    return assemble_findings(
        generated_at="fixed-time",
        ingested={},
        inputs=[],
        resolved=[],
        profiles={},
        upstream=upstream,
        psa_records=[],
        ranked=ranked,
        convergence={"list_3": {"funnel": {"after_severity_filter": len(ranked)}}},
        coverage={"list_3": {"assessed": len(upstream)}},
        headline_portfolio=headline,
        limit=None,
        settings=settings,
        ontology=synthetic_ontology(),
    )


def test_parser_preserves_routes_multiplicity_and_is_deterministic() -> None:
    # Parsing keeps duplicate path observations, is repeatable and never modifies the input.
    records = [record(ids=ids) for ids in (("node",), ("u2", "node"), ("u2", "u3", "node"))]
    records.append(deepcopy(records[-1]))
    data = data_for(records)
    before = deepcopy(data)
    first = fetch_result(data).paths
    assert first == fetch_result(data).paths
    assert data == before
    assert [len(path.hops) for path in first] == [1, 2, 3, 3]
    assert [path.path_index for path in first] == [0, 1, 2, 3]
    assert [hop.tier for hop in first[2].hops] == [2, 3, 4]
    for path, raw in zip(first, records, strict=True):
        assert path.source_entity_id == raw["source_entity_id"] == "s1"
        assert path.source_entity_id not in [hop.entity_id for hop in path.hops]
        assert [hop.model_dump() for hop in path.hops] == raw["path"]
    assert first[2].hops == first[3].hops
    assert first[2].hops[0].components[0] == first[2].hops[0].components[2]


@pytest.mark.parametrize("optional", ["absent", "null", "empty"])
@pytest.mark.parametrize("partial,has_entities", [(False, True), (True, True), (False, False)])
def test_optional_paths_do_not_change_status(
    settings: Settings, optional: str, partial: bool, has_entities: bool
) -> None:
    # Missing path detail never changes coverage beyond what the response contract supports.
    data = data_for([record()]) if has_entities else {"entities": {}}
    data.pop("paths", None)
    if optional != "absent":
        data["paths"] = None if optional == "null" else []
    result = fetch_result(data, partial=partial)
    assert result.paths == []
    if optional in ("absent", "null"):
        # The SDK requires a paths list, so a missing or null one fails closed.
        assert result.status == "error"
        assert result.error_type == "ValidationError"
        assert result.entities == {}
    else:
        assert result.status == (
            "partial" if partial else "assessed" if has_entities else "no_data"
        )
        assert result.error_type is None
        assert bool(result.entities) == has_entities


@pytest.mark.parametrize("omit_optional", [False, True])
def test_null_optional_fields_empty_countries_and_components(omit_optional: bool) -> None:
    # A null or missing required path field rejects the supplier's entire result.
    raw = record()
    empty = component(None)
    empty.update(departure_countries=[], arrival_countries=[], min_date=None, max_date=None)
    if omit_optional:
        for key in ("hs_code", "min_date", "max_date"):
            empty.pop(key)
    raw["path"][0]["components"] = [empty]
    raw["path"][1]["components"] = []
    data = data_for([raw])
    # The SDK requires hs_code, and it can't be null.
    failed = fetch_result(data)
    assert failed.status == "error"
    assert failed.error_type == "ValidationError"
    assert failed.entities == {} and failed.paths == []


@pytest.mark.parametrize("omit_dates", [False, True])
def test_optional_dates_empty_countries_and_components(omit_dates: bool) -> None:
    # Missing dates and empty component lists are allowed and kept exactly as received.
    raw = record()
    empty = component("8542")
    empty.update(departure_countries=[], arrival_countries=[], min_date=None, max_date=None)
    if omit_dates:
        empty.pop("min_date")
        empty.pop("max_date")
    raw["path"][0]["components"] = [empty]
    raw["path"][1]["components"] = []
    result = fetch_result(data_for([raw]))
    assert result.status == "assessed" and result.error_type is None
    path = result.paths[0]
    assert path.hops[0].components[0].model_dump() == {
        "hs_code": "8542",
        "departure_countries": [],
        "arrival_countries": [],
        "min_date": None,
        "max_date": None,
    }
    assert path.hops[1].components == []


@pytest.mark.parametrize(
    "target,key,value",
    [
        *[("data", "paths", value) for value in cast(tuple[object, ...], ({}, "", 0, False, ()))],
        *[("record_entry", "", value) for value in cast(tuple[object, ...], (None, [], "path"))],
        *[
            ("record", "source_entity_id", value)
            for value in cast(tuple[object, ...], (_MISSING, None, 7, "", " \t"))
        ],
        *[
            ("record", "path", value)
            for value in cast(tuple[object, ...], (_MISSING, None, {}, (), []))
        ],
        *[("hop_entry", "", value) for value in cast(tuple[object, ...], (None, [], "hop"))],
        *[
            ("hop", "tier", value)
            for value in cast(tuple[object, ...], (_MISSING, None, "2", 2.0, True))
        ],
        *[
            ("hop", "entity_id", value)
            for value in cast(tuple[object, ...], (_MISSING, None, 1, "", " \t", "absent"))
        ],
        *[
            ("hop", "components", value)
            for value in cast(tuple[object, ...], (_MISSING, None, {}, ()))
        ],
        *[
            ("component_entry", "", value)
            for value in cast(tuple[object, ...], (None, [], "component"))
        ],
        *[
            ("component", key, value)
            for key in ("departure_countries", "arrival_countries")
            for value in cast(tuple[object, ...], (_MISSING, None, "USA", ("USA",), [1], [True]))
        ],
        *[
            ("component", key, value)
            for key in ("hs_code", "min_date", "max_date")
            for value in cast(tuple[object, ...], (1, True, [], {}))
        ],
    ],
)
def test_malformed_paths_raise_and_isolate_the_supplier(
    settings: Settings, target: str, key: str, value: object
) -> None:
    # A malformed path fails only its own supplier, and no partial path list is kept.
    data = data_for([record(), record()])
    raw = data["paths"][1]
    hop = raw["path"][0]
    if target == "record_entry":
        data["paths"][1] = value
    elif target == "hop_entry":
        raw["path"][0] = value
    elif target == "component_entry":
        hop["components"][0] = value
    else:
        selected = {"data": data, "record": raw, "hop": hop, "component": hop["components"][0]}[
            target
        ]
        if value is _MISSING:
            selected.pop(key)
        else:
            selected[key] = value
    if isinstance(value, tuple):
        # Tuples cannot pass through JSON, so check them strictly at the Python boundary instead of
        # serializing them into valid lists.
        with pytest.raises(ValueError):
            if target == "data":
                UpstreamResult.model_validate(
                    {
                        "supplier_id": "s1",
                        "entities": {},
                        "paths": value,
                        "partial_results": True,
                        "explored_count": 12,
                        "status": "partial",
                    }
                )
            elif target == "record":
                SupplyPath.model_validate(
                    {
                        "source_entity_id": "s1",
                        "path_index": 0,
                        "hops": value,
                    }
                )
            elif target == "hop":
                SupplyPathHop.model_validate(hop)
            else:
                PathComponent.model_validate(hop["components"][0])
    else:
        failed = fetch_result(data, partial=True, explored=12)
        # The retired parser raised here; the same check now runs on the SDK result instead.
        assert failed.error_type == "ValidationError"
        assert failed.status == "error"
        assert failed.entities == {} and failed.paths == []
        assert failed.partial_results is True and failed.explored_count == 12


def test_source_order_and_tier_annotation_are_independent_of_hop_position() -> None:
    # Sayari's tier is kept as received, whatever the hop's list position or local graph distance.
    raw = record(ids=("z", "a", "node"))
    for hop, tier in zip(raw["path"], (7, -2, 4), strict=True):
        hop["tier"] = tier
    data = data_for([raw])
    path = fetch_result(data).paths[0]
    assert [(hop.entity_id, hop.tier) for hop in path.hops] == [("z", 7), ("a", -2), ("node", 4)]
    direct = SupplyPath(
        source_entity_id="s1",
        path_index=17,
        hops=[SupplyPathHop(tier=7, entity_id="z", components=[])],
    )
    assert direct.path_index == 17 and direct.hops[0].tier == 7
    with pytest.raises(ValueError):
        SupplyPath.model_validate({**direct.model_dump(), "path_index": -1})
    with pytest.raises(ValueError):
        SupplyPath.model_validate({**direct.model_dump(), "path_index": True})
    with pytest.raises(ValueError):
        SupplyPathHop.model_validate({"tier": True, "entity_id": "node", "components": []})
    with pytest.raises(ValueError):
        PathComponent.model_validate({"departure_countries": ("USA",), "arrival_countries": []})


def test_supply_paths_schema_exact_types_nullability_and_primary_key() -> None:
    # The path table enforces its declared types, its nullability and its primary key.
    with closing(duckdb.connect(":memory:")) as con:
        create_schema(con)
        assert {row[0] for row in con.execute("SHOW TABLES").fetchall()} == {
            "suppliers",
            "upstream_entities",
            "supplier_upstream",
            "risk_factors",
            "supply_paths",
        }
        assert con.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name='supply_paths' ORDER BY ordinal_position"
        ).fetchall() == [
            ("source_entity_id", "VARCHAR", "NO"),
            ("path_index", "INTEGER", "NO"),
            ("hop_position", "INTEGER", "NO"),
            ("tier", "INTEGER", "NO"),
            ("entity_id", "VARCHAR", "NO"),
            ("terminal_entity_id", "VARCHAR", "NO"),
            (
                "components",
                "STRUCT(hs_code VARCHAR, departure_countries VARCHAR[], "
                "arrival_countries VARCHAR[], min_date VARCHAR, max_date VARCHAR)[]",
                "NO",
            ),
        ]
        con.execute("INSERT INTO supply_paths VALUES ('s1', 2, 0, 8, 'node', 'node', [])")
        with pytest.raises(duckdb.ConstraintException):
            con.execute("INSERT INTO supply_paths VALUES ('s1', 2, 0, 9, 'other', 'other', [])")


def test_warehouse_round_trip_all_paths_order_components_and_rebuild(tmp_path: Path) -> None:
    # Rebuilding the warehouse keeps every hop in order along with its full set of components.
    records = [record("s1", ("node",)), record("s1", ("u2", "node")), record()]
    records[1]["path"][0]["components"] = []
    records[2]["path"][0]["tier"] = 8
    records[2]["path"][1]["components"][1].update(
        departure_countries=[], arrival_countries=[], min_date=None, max_date=None
    )
    upstream = {"s1": retrieval("s1", records * 9), "s2": retrieval("s2", [record("s2")])}
    before = deepcopy(upstream)
    rows = [supplier("s1"), supplier("s2", 3)]
    path = tmp_path / "paths.duckdb"
    expected = [
        (
            source,
            route.path_index,
            position,
            hop.tier,
            hop.entity_id,
            route.hops[-1].entity_id,
            [item.model_dump() for item in hop.components],
        )
        for source, result in upstream.items()
        for route in result.paths
        for position, hop in enumerate(route.hops)
    ]
    for _ in range(2):
        with closing(build_warehouse(path, rows, {}, upstream)) as con:
            actual = con.execute(
                "SELECT * FROM supply_paths ORDER BY source_entity_id, path_index, hop_position"
            ).fetchall()
            assert actual == expected
            assert len(actual) == 57
    assert upstream == before


@pytest.mark.parametrize("failure", ["no_data", "error", "wrong_source", "duplicate_key"])
def test_invalid_path_load_rolls_back_the_existing_warehouse(tmp_path: Path, failure: str) -> None:
    # A failed path load rolls back, leaving the previous valid warehouse in place.
    path = tmp_path / "atomic.duckdb"
    good = retrieval("s1", [record()])
    with closing(build_warehouse(path, [supplier("s1")], {}, {"s1": good})) as con:
        before = con.execute("SELECT * FROM supply_paths ORDER BY hop_position").fetchall()
    bad = good.model_copy(deep=True)
    expected_error: type[Exception] = ValueError
    if failure in {"no_data", "error"}:
        cast(Any, bad).status = failure
        bad.entities = {}
    elif failure == "wrong_source":
        bad.paths[0].source_entity_id = "s2"
    else:
        bad.paths.append(bad.paths[0].model_copy(deep=True))
        expected_error = duckdb.ConstraintException
    with pytest.raises(expected_error):
        build_warehouse(path, [supplier("s1")], {}, {"s1": bad})
    with closing(duckdb.connect(str(path))) as con:
        assert con.execute("SELECT * FROM supply_paths ORDER BY hop_position").fetchall() == before


def test_round_robin_preserves_suppliers_whole_paths_and_all_evidence(settings: Settings) -> None:
    # The capped examples take turns across suppliers, and every selected path is stored whole.
    many = [record("a") for _ in range(25)]
    many[0]["path"][0]["components"] = [component("1111")] * 19
    upstream = {
        "c": retrieval("c", [record("c")]),
        "b": retrieval("b", [record("b"), record("b")]),
        "a": retrieval("a", many),
    }
    ranked = [ranked_node("node", ["c", "a", "b", "missing"])]
    before = deepcopy(upstream)
    first = assemble(settings, upstream, ranked)
    second = assemble(settings, dict(reversed(list(upstream.items()))), ranked)
    assert first.model_dump_json() == second.model_dump_json()
    node = cast(dict[str, Any], first.paths)["nodes"]["node"]
    assert node["retrieved_path_count"] == 28
    assert node["stored_path_count"] == PATH_MAX_PER_NODE == 6
    assert node["truncated"] is True
    assert node["path_observed_supplier_ids"] == ["a", "b", "c"]
    assert [path["source_entity_id"] for path in node["paths"]] == ["a", "b", "c", "a", "b", "a"]
    flat_first_n = [route for source in sorted(upstream) for route in upstream[source].paths][
        :PATH_MAX_PER_NODE
    ]
    assert {route.source_entity_id for route in flat_first_n} == {"a"}
    assert {route["source_entity_id"] for route in node["paths"]} == {"a", "b", "c"}
    for stored in node["paths"]:
        original = upstream[stored["source_entity_id"]].paths[stored["path_index"]]
        assert stored == original.model_dump()
        assert len(stored["hops"]) == 3
    assert len(node["paths"][0]["hops"][0]["components"]) == 19
    assert upstream == before
    connected = {
        row["entity_id"] for row in cast(list[dict[str, Any]], first.shared_nodes)[0]["suppliers"]
    }
    assert connected - set(node["path_observed_supplier_ids"]) == {"missing"}


def test_full_observed_set_survives_more_suppliers_than_cap(settings: Settings) -> None:
    # Suppliers left out by the cap still appear in the list of observed sources.
    sources = [f"s{i}" for i in range(8)]
    upstream = {source: retrieval(source, [record(source)]) for source in reversed(sources)}
    result = assemble(settings, upstream, [ranked_node("node", list(reversed(sources)))])
    node = cast(dict[str, Any], result.paths)["nodes"]["node"]
    assert node["retrieved_path_count"] == 8 and node["stored_path_count"] == 6
    assert node["truncated"] is True
    assert node["path_observed_supplier_ids"] == sources
    assert [path["source_entity_id"] for path in node["paths"]] == sources[:6]
    assert set(node["path_observed_supplier_ids"]) - {
        path["source_entity_id"] for path in node["paths"]
    } == {"s6", "s7"}


def test_path_group_order_uses_hops_then_canonical_components_then_index(
    settings: Settings,
) -> None:
    # Example paths are ordered by hop count, then components, then source index.
    records = [record(ids=("z", "node")), record(ids=("a", "node"))]
    for hs_code in ("z", "\u00e9", None, None):
        raw = record(ids=("node",))
        raw["path"][0]["components"] = [component(hs_code if hs_code is not None else "8542")]
        records.append(raw)
    upstream = {"s1": retrieval("s1", records)}
    # The domain model allows a null hs_code even though the SDK requires a string.
    for index in (4, 5):
        upstream["s1"].paths[index].hops[0].components[0].hs_code = None
    upstream["s1"].paths.reverse()
    result = assemble(settings, upstream, [ranked_node("node", ["s1"])])
    node = cast(dict[str, Any], result.paths)["nodes"]["node"]
    assert [path["path_index"] for path in node["paths"]] == [1, 2, 3, 4, 5, 0]
    assert node["truncated"] is False
    assert node["retrieved_path_count"] == node["stored_path_count"] == 6


def test_findings_projection_key_order_labels_and_portfolio_scope(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The path block keeps a stable key order and its labels, and covers only the headline scope.
    upstream = {
        "s1": retrieval("s1", [record(), record(ids=("node", "tail"))]),
        "outside": retrieval("outside", [record("outside")]),
    }
    upstream["outside"].entities["node"].label = "Wrong response"
    upstream["s1"].entities["u2"].translated_label = "Translated intermediate"
    upstream["s1"].entities["u3"].label = None
    ranked = [
        ranked_node("z-zero", ["s1"]),
        ranked_node("node", ["s1"]),
        ranked_node("tail", ["outside"], "another_portfolio"),
        ranked_node("u2", ["s1"]),
    ]
    guard = Mock(side_effect=AssertionError("Findings must use parsed models without DuckDB"))
    monkeypatch.setattr(duckdb, "connect", guard)
    result = assemble(settings, upstream, ranked)
    data = json.loads(result.model_dump_json())
    assert list(data) == OLD_FINDINGS_KEYS + ["paths", "ontology"]
    assert set(data) - set(OLD_FINDINGS_KEYS) == {"paths", "ontology"}
    assert list(data["manifest"]) == MANIFEST_KEYS
    assert data["manifest"]["path_max_per_node"] == 6
    block = data["paths"]
    assert list(block) == ["entities", "nodes"]
    assert list(block["nodes"]) == ["z-zero", "node", "u2"]
    assert list(block["entities"]) == ["node", "tail", "u2", "u3"]
    assert block["entities"]["node"]["label"] == upstream["s1"].entities["node"].label
    assert block["entities"]["u2"]["translated_label"] == "Translated intermediate"
    assert block["entities"]["u3"]["label"] is None
    for entity in block["entities"].values():
        assert list(entity) == ["label", "translated_label"]
    hop_ids = set()
    for node in block["nodes"].values():
        assert list(node) == [
            "retrieved_path_count",
            "stored_path_count",
            "truncated",
            "path_observed_supplier_ids",
            "paths",
        ]
        assert node["truncated"] == (node["stored_path_count"] < node["retrieved_path_count"])
        for path in node["paths"]:
            assert list(path) == ["source_entity_id", "path_index", "hops"]
            for hop in path["hops"]:
                hop_ids.add(hop["entity_id"])
                assert list(hop) == ["tier", "entity_id", "components"]
                for item in hop["components"]:
                    assert list(item) == COMPONENT_KEYS
    assert set(block["entities"]) == hop_ids
    # No retrieved path touches z-zero at any position, so it holds no evidence.
    assert block["nodes"]["z-zero"] == {
        "retrieved_path_count": 0,
        "stored_path_count": 0,
        "truncated": False,
        "path_observed_supplier_ids": [],
        "paths": [],
    }
    # u2 is traversed on the way further upstream, which now counts as path evidence.
    assert block["nodes"]["u2"]["retrieved_path_count"] == 1
    assert block["nodes"]["u2"]["path_observed_supplier_ids"] == ["s1"]
    # Both stored records reach node: one ends there, one passes through it.
    assert block["nodes"]["node"]["retrieved_path_count"] == 2
    assert assemble(settings, upstream, ranked, headline=None).paths == {
        "entities": {},
        "nodes": {},
    }
    guard.assert_not_called()


def test_offline_artifacts_protected_blocks_and_repeat_replay(
    graph: tuple[Settings, ResponseCache], tmp_path: Path, forbid_external_stages: None
) -> None:
    # Replaying again with more cached paths changes only the path evidence; the rest is unchanged.
    configured, cache = graph
    output = tmp_path / "out"
    before = pipeline.run_pipeline(configured, all_sheets=True, offline=True, output_dir=output)
    before_html = (output / "report.html").read_text(encoding="utf-8")
    before_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    for i in range(1, 7):
        source = f"s{i}"
        key = cache.key(
            httpx.Request(
                "GET",
                configured.sayari_api_base + f"/v1/supply_chain/upstream/{source}",
                params={
                    "max_depth": configured.max_upstream_depth,
                    "limit": configured.upstream_limit,
                },
            ),
        )
        raw = json.loads(cache.get(key) or b"null")
        _upstream(
            cache,
            configured,
            source,
            raw["data"]["entities"],
            partial=raw["partial_results"],
            paths=[record(source, ("n-top",)) for _ in range(25 if i == 1 else 1)],
        )
    after = pipeline.run_pipeline(configured, all_sheets=True, offline=True, output_dir=output)
    for key in ("suppliers", "shared_nodes", "convergence", "coverage"):
        assert getattr(after, key) == getattr(before, key)
    assert after.model_dump(exclude={"paths", "generated_at"}) == before.model_dump(
        exclude={"paths", "generated_at"}
    )
    assert set(after.model_dump()) - set(OLD_FINDINGS_KEYS) == {"paths", "ontology"}
    old_manifest_keys = set(MANIFEST_KEYS) - {"path_max_per_node"}
    assert set(cast(dict[str, Any], after.manifest)) - old_manifest_keys == {"path_max_per_node"}
    # The report shows the formatted date, not the raw timestamp. Mask both, or this comparison
    # fails whenever the two runs fall in different minutes.
    after_html = (
        (output / "report.html")
        .read_text(encoding="utf-8")
        .replace(after.generated_at, "<timestamp>")
        .replace(report_datetime(after.generated_at), "<timestamp>")
    )
    before_html = before_html.replace(before.generated_at, "<timestamp>").replace(
        report_datetime(before.generated_at), "<timestamp>"
    )
    assert (
        before_html.partition('<section id="suppliers"')[0]
        == after_html.partition('<section id="suppliers"')[0]
    )
    # Only the suppliers section renders path evidence, so everything from the next section on must
    # be byte-identical. The bound has to be the first section after suppliers: partitioning at
    # convergence would now swallow suppliers itself, because convergence sits above it, and
    # partitioning at methodology would leave the flagged-entity list unchecked.
    assert (
        before_html.partition('<section id="flagged-entities"')[2]
        == after_html.partition('<section id="flagged-entities"')[2]
    )
    assert before_html != after_html
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["bounds"]
        == before_manifest["bounds"]
        == {
            key: getattr(configured, key)
            for key in ("hub_max_countries", "max_upstream_depth", "upstream_limit")
        }
    )
    assert manifest["api"]["data_http_attempts"] == 0
    assert cast(dict[str, Any], after.paths)["nodes"]["n-top"]["retrieved_path_count"] == 30
    assert cast(dict[str, Any], after.paths)["nodes"]["n-top"]["stored_path_count"] == 6
    assert cast(dict[str, Any], after.paths)["nodes"]["n-boundary"]["retrieved_path_count"] == 0
    artifacts = ["findings.json", "report.html"]
    first_bytes = [(output / name).read_bytes() for name in artifacts]
    repeated = pipeline.run_pipeline(configured, all_sheets=True, offline=True, output_dir=output)
    assert repeated == after
    assert [(output / name).read_bytes() for name in artifacts] == first_bytes
    assert (
        json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["api"][
            "data_http_attempts"
        ]
        == 0
    )

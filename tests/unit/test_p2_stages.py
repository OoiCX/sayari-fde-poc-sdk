"""P2 stages exercised through the SDK using synthetic offline response bytes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import httpx
import pytest

from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.enrich import fetch_profiles
from sayari_poc.models import InputEntity, ResolvedEntity
from sayari_poc.resolve import resolve_entities
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import (
    BudgetExceeded,
    CallState,
    OfflineCacheMiss,
    SayariAuthError,
    SayariError,
    SayariNotFound,
    SayariRateLimitError,
)
from sayari_poc.upstream import fetch_upstream, fetch_upstreams


def profile(entity_id: str = "root") -> dict[str, Any]:
    return {
        "id": entity_id,
        "label": "Synthetic profile",
        "degree": 0,
        "closed": False,
        "entity_url": "",
        "pep": False,
        "psa_count": 0,
        "sanctioned": False,
        "type": "company",
        "identifiers": [],
        "countries": ["SGP"],
        "source_count": {},
        "addresses": [],
        "trade_count": {},
        "relationship_count": {},
        "user_relationship_count": {},
        "attribute_count": {},
        "user_attribute_count": {},
        "related_entities_count": 0,
        "user_related_entities_count": 0,
        "user_record_count": 0,
        "risk": {},
    }


def upstream_payload(supplier_id: str = "root") -> dict[str, Any]:
    return {
        "filters": {},
        "partial_results": False,
        "explored_count": 1,
        "data": {
            "entities": {
                "node": {
                    "id": "node",
                    "type": "company",
                    "label": "Synthetic node",
                    "countries": ["USA", "SGP"],
                    "risk_factors": [],
                    "translated_label": "Synthetic translation",
                },
            },
            "paths": [
                {
                    "source_entity_id": supplier_id,
                    "path": [
                        {
                            "entity_id": "node",
                            "tier": 3,
                            "components": [
                                {
                                    "hs_code": "1234",
                                    "arrival_countries": ["SGP", "USA"],
                                    "departure_countries": ["CHN"],
                                    "min_date": "2020-01-01",
                                },
                            ],
                        },
                    ],
                }
            ],
        },
    }


def candidate(entity_id: str, strength: str = "strong", score: float = 130.5) -> dict[str, Any]:
    return {
        "profile": "corporate",
        "score": score,
        "entity_id": entity_id,
        "label": "Synthetic",
        "type": "company",
        "identifiers": [],
        "addresses": [],
        "countries": [],
        "sources": [],
        "typed_matched_queries": [],
        "matched_queries": [],
        "highlight": {},
        "explanation": {},
        "match_strength": {"value": strength},
    }


class QueuedCache(ResponseCache):
    def __init__(self, directory: Path, payloads: tuple[object, ...]) -> None:
        super().__init__(directory)
        self.payloads = iter(payloads)
        self.keys: list[str] = []

    def get(self, key: str) -> bytes | None:
        self.keys.append(key)
        payload = next(self.payloads, None)
        if isinstance(payload, BaseException):
            raise payload
        return json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def client_for(tmp_path: Path, *payloads: object) -> tuple[SayariClient, QueuedCache]:
    settings = Settings(_env_file=None, cache_dir=tmp_path / "cache")
    cache = QueuedCache(settings.cache_dir, payloads)
    return SayariClient(settings, cache, offline=True), cache


def source(number: int = 2) -> InputEntity:
    return InputEntity(
        name="Synthetic supplier", address=None, country=None, sheet="list_3", row_number=number
    )


def resolved(entity_id: str = "root", **updates: object) -> ResolvedEntity:
    return ResolvedEntity.model_validate(
        {
            "row_number": 2,
            "sheet": "list_3",
            "input_name": "Synthetic supplier",
            "entity_id": entity_id,
            "status": "resolved",
            **updates,
        }
    )


def test_rows_candidates_diagnostics_and_raw_scores_remain_in_source_order(tmp_path: Path) -> None:
    # Resolution keeps rows and candidates in source order and never rescales the relevance score.
    inputs = [source(8), source(2), source(2), source(19), source(3)]
    before = [row.model_dump_json() for row in inputs]
    payloads = [
        {"fields": {}, "data": [candidate("first", score=88.5), candidate("other", score=999.0)]},
        {
            "fields": {},
            "data": [candidate("weak", "weak"), candidate("bad", "future"), candidate("third")],
        },
        {"fields": {}, "data": []},
        RuntimeError("synthetic-sensitive-detail"),
        {"fields": {}, "data": [candidate("last", score=272.6)]},
    ]
    client, cache = client_for(tmp_path, *payloads)
    with client:
        rows = resolve_entities(client, inputs)
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 0
    assert [(r.row_number, r.sheet, r.input_name) for r in rows] == [
        (r.row_number, r.sheet, r.name) for r in inputs
    ]
    assert [r.status for r in rows] == ["resolved", "weak", "no_match", "error", "resolved"]
    assert rows[0].entity_id == "first" and rows[0].score == 88.5
    assert [c.entity_id for c in rows[0].candidates] == ["first", "other"]
    assert rows[1].candidate_count == 3
    assert [c.entity_id for c in rows[1].candidates] == ["weak", "third"]
    assert rows[1].candidate_errors == {1: "Malformed resolution candidate"}
    assert rows[2].entity_id is None and rows[2].candidates == []
    assert rows[3].error_type == "RuntimeError"
    assert "synthetic-sensitive-detail" not in rows[3].model_dump_json()
    assert rows[4].score == 272.6
    assert len(cache.keys) == len(inputs)
    assert [row.model_dump_json() for row in inputs] == before


@pytest.mark.parametrize("countries", [["USA", "JPN", "USA"], []])
def test_resolution_retains_sdk_candidate_countries_in_received_order(
    tmp_path: Path, countries: list[str]
) -> None:
    # A candidate's country list keeps the order and duplicates it was received with.
    first = candidate("first", "weak")
    first["countries"] = countries
    client, _ = client_for(tmp_path, {"fields": {}, "data": [first, candidate("second")]})
    with client:
        row = client.resolve(source(2))
    assert row.status == "weak" and row.entity_id == "first"
    assert row.candidates[0].countries == countries
    assert row.candidates[0].model_dump(mode="json")["countries"] == countries
    assert row.candidates[1].countries == []


@pytest.mark.parametrize("strength", ["", "STRONG", "medium", "unknown"])
def test_unknown_primary_strength_never_promotes_a_strong_alternate(
    tmp_path: Path, strength: str
) -> None:
    # When the top candidate is not acceptable, a strong alternate never takes its place.
    client, _ = client_for(
        tmp_path, {"fields": {}, "data": [candidate("first", strength), candidate("alternate")]}
    )
    with client:
        row = resolve_entities(client, [source()])[0]
    assert row.status == "error" and row.entity_id is None
    assert row.candidate_count == 2 and row.error_type == "ValidationError"


@pytest.mark.parametrize("payload", [{}, {"data": None}, {"data": {}}, []])
def test_bad_resolution_is_error_and_later_row_still_succeeds(
    tmp_path: Path, payload: object
) -> None:
    # A malformed resolution response fails only its own row; later rows still resolve.
    client, _ = client_for(tmp_path, payload, {"fields": {}, "data": []})
    with client:
        rows = resolve_entities(client, [source(), source(3)])
    assert [r.status for r in rows] == ["error", "no_match"]
    assert rows[0].error_type == "ValidationError"


def test_exact_resolution_query_preserves_unicode_optional_fields(tmp_path: Path) -> None:
    # The resolution query the SDK builds keeps Unicode text and the optional fields.
    row = source().model_copy(
        update={"name": "M?ller & S?hne", "address": "Stra?e 1", "country": "DEU"}
    )
    client, cache = client_for(tmp_path, {"fields": {}, "data": []})
    with client:
        resolve_entities(client, [row])
    request = httpx.Request(
        "GET",
        "https://api.sayari.com/v1/resolution",
        params={"name": row.name, "address": row.address, "country": row.country},
    )
    assert cache.keys == [cache.key(request)]


def test_only_resolved_rows_are_profiled_and_traversed_in_sorted_distinct_order(
    tmp_path: Path,
) -> None:
    # Only accepted identities are profiled and traversed, and each distinct ID only once.
    rows = [
        resolved("z"),
        resolved("a", status="weak"),
        resolved("a"),
        resolved("z"),
        resolved("unused", status="no_match"),
        resolved("failed", status="error"),
    ]
    before = [row.model_dump_json() for row in rows]
    client, cache = client_for(
        tmp_path, profile("z"), profile("a"), upstream_payload("a"), upstream_payload("z")
    )
    settings = Settings(_env_file=None, max_upstream_depth=4, upstream_limit=7)
    with client:
        profiles = fetch_profiles(client, rows)
        upstream = fetch_upstreams(client, rows, settings)
    assert list(profiles) == ["z", "a"]
    assert list(upstream) == ["a", "z"]
    assert [row.model_dump_json() for row in rows] == before
    requests = [
        httpx.Request("GET", "https://api.sayari.com/v1/entity_summary/" + entity_id)
        for entity_id in ("z", "a")
    ] + [
        httpx.Request(
            "GET",
            "https://api.sayari.com/v1/supply_chain/upstream/" + entity_id,
            params={"max_depth": 4, "limit": 7},
        )
        for entity_id in ("a", "z")
    ]
    assert cache.keys == [cache.key(request) for request in requests]


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded,
        OfflineCacheMiss,
        SayariAuthError,
        SayariNotFound,
        SayariRateLimitError,
        SayariError,
        RuntimeError,
    ],
)
def test_profile_failures_remain_per_row_and_successful_duplicate_clears_errors(
    tmp_path: Path, error: type[Exception], caplog: pytest.LogCaptureFixture
) -> None:
    # A later successful fetch for a duplicate identity clears its earlier profile error.
    rows = [resolved(), resolved("second"), resolved(row_number=9), resolved(status="weak")]
    before = [row.model_dump(exclude={"profile_error", "profile_error_type"}) for row in rows]
    client, cache = client_for(
        tmp_path, error("synthetic-sensitive-detail"), profile("second"), profile()
    )
    with client:
        profiles = fetch_profiles(client, rows)
    assert list(profiles) == ["second", "root"]
    assert len(cache.keys) == 3
    assert all(row.profile_error is row.profile_error_type is None for row in rows)
    assert [
        row.model_dump(exclude={"profile_error", "profile_error_type"}) for row in rows
    ] == before
    assert "synthetic-sensitive-detail" not in caplog.text


def test_profile_failed_duplicates_and_malformed_profile_never_disappear(tmp_path: Path) -> None:
    # Rows whose profile failed stay in the output, each with its own error category.
    rows = [resolved(), resolved(row_number=8), resolved("second"), resolved("last")]
    raw = profile("second")
    raw["countries"] = "synthetic-sensitive-detail"
    client, _ = client_for(
        tmp_path, RuntimeError("private"), BudgetExceeded("private"), raw, profile("last")
    )
    with client:
        profiles = fetch_profiles(client, rows)
    assert list(profiles) == ["last"]
    assert [row.profile_error_type for row in rows] == [
        "RuntimeError",
        "BudgetExceeded",
        "ValidationError",
        None,
    ]
    assert all(row.status == "resolved" for row in rows)
    assert all("private" not in row.model_dump_json() for row in rows)


@pytest.mark.parametrize("entity_id", [None, "", " "])
def test_profile_missing_id_is_validation_error_without_fetch(
    tmp_path: Path, entity_id: str | None
) -> None:
    # A missing canonical ID fails profile validation before the cache is touched.
    row = resolved()
    row.entity_id = entity_id
    client, cache = client_for(tmp_path)
    with client:
        assert fetch_profiles(client, [row]) == {}
    assert row.profile_error_type == "ValidationError" and cache.keys == []


@pytest.mark.parametrize("level", ["high", "elevated", "relevant"])
def test_profile_risk_evidence_and_severity_preserved(tmp_path: Path, level: str) -> None:
    # Normalizing a profile keeps its countries, zero values and the levels Sayari assigned.
    raw = profile()
    raw["translated_label"] = "Synthetic translated label"
    raw["countries"] = ["SGP", "USA", "SGP"]
    raw["risk"] = {
        "absent": {"value": False, "level": "critical", "metadata": {}},
        "present": {"value": True, "level": level, "metadata": {"nested": {"values": [2, 1]}}},
        "zero": {"value": 0, "level": "relevant", "metadata": {}},
    }
    client, _ = client_for(tmp_path, raw)
    with client:
        result = fetch_profiles(client, [resolved()])["root"]
    assert result.countries == ["SGP", "USA", "SGP"]
    assert result.translated_label == "Synthetic translated label"
    assert [factor.factor for factor in result.risk_factors] == ["present", "zero"]
    assert result.max_level == level
    assert result.risk_factors[0].metadata == {"nested": {"values": [2, 1]}}
    assert result.risk_factors[1].value == 0


@pytest.mark.parametrize(
    ("partial", "empty", "status"),
    [
        (False, False, "assessed"),
        (False, True, "no_data"),
        (True, False, "partial"),
        (True, True, "partial"),
    ],
)
def test_coverage_precedence_and_no_inference_from_limit_or_count(
    tmp_path: Path, partial: bool, empty: bool, status: str
) -> None:
    # Coverage follows the flags Sayari returns; completeness is never inferred from the counts.
    raw = upstream_payload()
    raw.update(partial_results=partial, explored_count=1000000)
    if empty:
        raw["data"] = {"entities": {}, "paths": []}
    client, _ = client_for(tmp_path, raw)
    with client:
        result = fetch_upstream(client, "root", Settings(_env_file=None, upstream_limit=1))
    assert result.status == status and result.partial_results is partial
    assert result.explored_count == 1000000 and result.error_type is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("partial_results", "false"),
        ("partial_results", 0),
        ("partial_results", None),
        ("explored_count", True),
        ("explored_count", 7.0),
        ("explored_count", "7"),
        ("explored_count", -1),
        ("explored_count", None),
    ],
)
def test_raw_coverage_fails_closed_and_valid_other_metadata_survives(
    tmp_path: Path, field: str, value: object
) -> None:
    # Invalid coverage metadata fails closed, and the other valid field is kept for diagnosis.
    raw = upstream_payload()
    raw.update(partial_results=True, explored_count=7)
    raw[field] = value
    client, _ = client_for(tmp_path, raw)
    with client:
        result = fetch_upstream(client, "root", Settings(_env_file=None))
    assert result.status == "error" and result.error_type == "ValidationError"
    assert result.entities == {} and result.paths == []
    assert result.partial_results is (False if field == "partial_results" else True)
    assert result.explored_count == (None if field == "explored_count" else 7)


@pytest.mark.parametrize(
    "malformation", ["entity", "path", "orphan", "failed", "component", "filters"]
)
def test_bad_upstream_retains_coverage_but_never_partial_parse(
    tmp_path: Path, malformation: str
) -> None:
    # A malformed traversal keeps its coverage metadata but never a partly parsed entity set.
    raw = upstream_payload("a")
    raw.update(partial_results=True, explored_count=77)
    if malformation == "entity":
        raw["data"]["entities"]["bad"] = {"id": "different"}
    elif malformation == "path":
        raw["data"]["paths"].append({"path": []})
    elif malformation == "orphan":
        raw["data"]["paths"][0]["path"][0]["entity_id"] = "missing"
    elif malformation == "failed":
        raw["success"] = False
    elif malformation == "component":
        raw["data"]["paths"][0]["path"][0]["components"].append({})
    else:
        del raw["filters"]
    client, _ = client_for(tmp_path, raw, upstream_payload("b"))
    with client:
        results = fetch_upstreams(client, [resolved("a"), resolved("b")], Settings(_env_file=None))
    failed = results["a"]
    assert failed.status == "error" and failed.error_type == "ValidationError"
    assert failed.partial_results is True and failed.explored_count == 77
    assert failed.entities == {} and failed.paths == []
    assert results["b"].status == "assessed"


def test_paths_preserve_all_order_duplicates_and_sayari_tiers(tmp_path: Path) -> None:
    # Paths keep their source order, duplicate observations and the tiers Sayari returned.
    raw = upstream_payload()
    hop = raw["data"]["paths"][0]["path"][0]
    component = hop["components"][0]
    component["arrival_countries"] = ["USA", "SGP", "USA"]
    component["departure_countries"] = ["CHN", "USA", "CHN"]
    hop["components"].extend([deepcopy(component), {**component, "hs_code": "5678"}])
    raw["data"]["paths"][0]["path"].append({**deepcopy(hop), "tier": 1})
    raw["data"]["paths"] *= 2
    client, _ = client_for(tmp_path, raw)
    with client:
        result = fetch_upstream(client, "root", Settings(_env_file=None))
    assert [path.path_index for path in result.paths] == [0, 1]
    assert result.paths[0].hops == result.paths[1].hops
    assert [hop.tier for hop in result.paths[0].hops] == [3, 1]
    components = result.paths[0].hops[0].components
    assert [part.hs_code for part in components] == ["1234", "1234", "5678"]
    assert components[0].arrival_countries == ["USA", "SGP", "USA"]
    assert components[0].departure_countries == ["CHN", "USA", "CHN"]
    assert components[0].min_date == "2020-01-01"


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded,
        OfflineCacheMiss,
        SayariAuthError,
        SayariNotFound,
        SayariRateLimitError,
        SayariError,
        RuntimeError,
    ],
)
def test_upstream_failure_is_never_successful_empty_coverage(
    tmp_path: Path, error: type[Exception]
) -> None:
    # A failed traversal stays an error (never no_data), and later suppliers still succeed.
    client, _ = client_for(tmp_path, error("synthetic-sensitive-detail"), upstream_payload("b"))
    with client:
        results = fetch_upstreams(client, [resolved("a"), resolved("b")], Settings(_env_file=None))
    assert results["a"].status == "error" and results["a"].error_type == error.__name__
    assert results["a"].partial_results is False and results["a"].explored_count is None
    assert results["b"].status == "assessed"
    assert "synthetic-sensitive-detail" not in results["a"].model_dump_json()


@pytest.mark.parametrize("stage", ["resolve", "enrich", "upstream"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate_from_every_stage(
    tmp_path: Path, stage: str, error: type[BaseException]
) -> None:
    # No stage swallows KeyboardInterrupt or SystemExit.
    client, _ = client_for(tmp_path, error())
    with client, pytest.raises(error):
        if stage == "resolve":
            resolve_entities(client, [source()])
        elif stage == "enrich":
            fetch_profiles(client, [resolved()])
        else:
            fetch_upstreams(client, [resolved()], Settings(_env_file=None))


def test_stage_boundary_unexpected_failure_keeps_row_and_safe_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An unexpected stage error keeps the row and never exposes the exception's message.
    client = create_autospec(SayariClient, instance=True, spec_set=True)
    client.resolve.side_effect = [TypeError("synthetic-sensitive-detail"), resolved()]
    rows = resolve_entities(client, [source(7), source()])
    assert len(rows) == 2 and rows[0].error_type == "TypeError"
    assert rows[1].status == "resolved"
    client.upstream.side_effect = [TypeError("synthetic-sensitive-detail"), RuntimeError("private")]
    upstream = fetch_upstreams(client, [resolved("a"), resolved("b")], Settings(_env_file=None))
    assert [result.error_type for result in upstream.values()] == ["TypeError", "RuntimeError"]
    assert all(result.status == "error" for result in upstream.values())
    assert "synthetic-sensitive-detail" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_empty_batches_make_no_calls(tmp_path: Path) -> None:
    # An empty input gives an empty result from every stage, with no retrieval at all.
    client, cache = client_for(tmp_path)
    with client:
        assert resolve_entities(client, []) == []
        assert fetch_profiles(client, []) == {}
        assert fetch_upstreams(client, [], Settings(_env_file=None)) == {}
    assert cache.keys == []


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded,
        OfflineCacheMiss,
        SayariAuthError,
        SayariNotFound,
        SayariRateLimitError,
        SayariError,
        RuntimeError,
    ],
)
def test_resolution_failure_types_do_not_expose_messages(
    tmp_path: Path, error: type[Exception]
) -> None:
    # A resolution failure records its error type but none of the exception's sensitive text.
    client, _ = client_for(
        tmp_path, error("synthetic-sensitive-detail"), {"fields": {}, "data": []}
    )
    with client:
        rows = resolve_entities(client, [source(), source(3)])
    assert [row.status for row in rows] == ["error", "no_match"]
    assert rows[0].error_type == error.__name__
    assert "synthetic-sensitive-detail" not in rows[0].model_dump_json()


@pytest.mark.parametrize("entity_id", ["", " ", "bad/id"])
def test_invalid_resolved_upstream_id_has_error_result_without_io(
    tmp_path: Path, entity_id: str
) -> None:
    # An unsafe supplier ID becomes a traversal error before any cache or network access.
    client, cache = client_for(tmp_path)
    with client:
        result = fetch_upstreams(client, [resolved(entity_id)], Settings(_env_file=None))[entity_id]
    assert result.status == "error" and result.error_type == "ValidationError"
    assert result.entities == {} and result.paths == []
    assert cache.keys == []


@pytest.mark.parametrize("value", [True, 0, 3, 9007199254740993, 2.0, 4.63, "documented-string"])
def test_approved_risk_scalar_preservation_keeps_value_and_received_type(
    tmp_path: Path, value: bool | int | float | str
) -> None:
    # The adapter keeps each approved scalar's value and its exact Python type.
    raw = profile()
    raw["risk"] = {"factor": {"value": value, "level": "high", "metadata": {}}}
    client, _ = client_for(tmp_path, raw)
    with client:
        result = fetch_profiles(client, [resolved()])["root"].risk_factors[0]
    assert result.value == value and type(result.value) is type(value)
    encoded = json.loads(result.model_dump_json())["value"]
    assert encoded == value and type(encoded) is type(value)


def test_raw_risk_scalars_cannot_leak_between_logical_calls_or_into_repr(tmp_path: Path) -> None:
    # Raw scalar values stay with the call that received them and never appear in a repr.
    first, second, third = profile("first"), profile("second"), profile("third")
    first["risk"] = {"factor": {"value": 3, "level": "high", "metadata": {}}}
    second["risk"] = {"factor": {"value": 3.0, "level": "high", "metadata": {}}}
    client, _ = client_for(tmp_path, first, second, third, upstream_payload())
    with client:
        profiles = fetch_profiles(
            client, [resolved("first"), resolved("second"), resolved("third")]
        )
        upstream = fetch_upstream(client, "root", Settings(_env_file=None))
    assert type(profiles["first"].risk_factors[0].value) is int
    assert type(profiles["second"].risk_factors[0].value) is float
    assert profiles["third"].risk_factors == []
    assert upstream.explored_count == 1 and upstream.status == "assessed"
    assert "synthetic-sensitive" not in repr(CallState(risk_values=(("synthetic-sensitive", 3),)))


@pytest.mark.parametrize("value", [[], {}, float("nan"), float("inf")])
def test_raw_risk_scalar_restoration_still_fails_strict_domain_validation(
    tmp_path: Path, value: object
) -> None:
    # Restoring a scalar's original type never lets it slip past strict profile validation.
    raw = profile()
    raw["risk"] = {"factor": {"value": value, "level": "high", "metadata": {}}}
    row = resolved()
    client, _ = client_for(tmp_path, raw, profile("later"))
    with client:
        profiles = fetch_profiles(client, [row, resolved("later")])
    assert row.profile_error_type == "ValidationError"
    assert list(profiles) == ["later"]


def test_mocked_response_and_cached_replay_preserve_scalar_types_and_raw_bytes(
    tmp_path: Path,
) -> None:
    # A mocked live fetch and the cached replay agree on the raw bytes and the scalar types.
    raw = profile()
    raw["risk"] = {
        "integer": {"value": 3, "level": "high", "metadata": {}},
        "float": {"value": 3.0, "level": "high", "metadata": {}},
    }
    content = json.dumps(raw, indent=2).encode()
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path / "cache",
        sayari_api_base="https://p2.invalid",
        sayari_client_id="synthetic-client-id",
        sayari_client_secret="synthetic-client-secret",
    )
    cache = ResponseCache(settings.cache_dir)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-access-token",
                    "expires_in": 86400,
                    "token_type": "Bearer",
                },
            )
        return httpx.Response(200, content=content, headers={"Content-Type": "application/json"})

    with SayariClient(settings, cache, transport=httpx.MockTransport(respond)) as client:
        live = fetch_profiles(client, [resolved()])["root"]
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 1
    assert len(list(settings.cache_dir.glob("*.json"))) == 1
    assert next(settings.cache_dir.glob("*.json")).read_bytes() == content
    with SayariClient(settings, cache, offline=True) as client:
        replay = fetch_profiles(client, [resolved()])["root"]
        assert client.audit.auth_http_attempts == client.audit.data_http_attempts == 0
        assert client.cache_hits == client.cache_lookups == 1
    assert live.model_dump_json() == replay.model_dump_json()
    assert type(replay.risk_factors[0].value) is int
    assert type(replay.risk_factors[1].value) is float

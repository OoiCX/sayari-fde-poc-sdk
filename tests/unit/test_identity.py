"""Identity findings from synthetic warehouses and retained profile evidence only."""

import json
from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from sayari_poc.identity import psa_exposure
from sayari_poc.models import (
    EntityProfile,
    Findings,
    ResolvedEntity,
    RiskFactor,
)
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import AuditedTransport
from tests.ontology_support import synthetic_ontology

PSA_COLUMNS = ["portfolio", "row_number", "entity_id", "psa_count", "psa_risky", "psa_status"]


@pytest.fixture(autouse=True)
def forbid_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("sayari_poc.identity.load_ontology", synthetic_ontology)
    guard = Mock(side_effect=AssertionError("Identity analysis must never access the API"))
    monkeypatch.setattr(SayariClient, "__init__", guard)
    monkeypatch.setattr(AuditedTransport, "handle_request", guard)
    yield
    guard.assert_not_called()


def supplier(
    entity_id: str | None = "S1", row: int = 1, portfolio: str = "list_3"
) -> ResolvedEntity:
    return ResolvedEntity(
        row_number=row,
        sheet=portfolio,
        input_name="\uae30\uc544",
        entity_id=entity_id,
        label="\uae30\uc544",
        translated_label="Kia",
        status="resolved",
    )


def profile(entity_id: str = "S1", count: int = 4, factors: tuple[str, ...] = ()) -> EntityProfile:
    return EntityProfile(
        entity_id=entity_id,
        label="\uae30\uc544",
        translated_label="Kia",
        countries=["KOR"],
        degree=999,
        psa_count=count,
        max_level="high",
        risk_factors=[RiskFactor(factor=factor, metadata={}) for factor in factors],
    )


@pytest.mark.parametrize(
    "count,factors,expected",
    [
        (0, (), False),
        (4, (), False),
        (999, (), False),
        (0, ("psa_ofac_50_percent_rule",), True),
        (4, ("PSA_OFAC_50_PERCENT_RULE",), None),
        (4, ("owned_by_psa_entity",), None),
        (4, ("psa",), None),
        (4, (" psA_ofac",), None),
        (4, ("uflpa",), False),
        (4, ("psa_misleading_name",), False),
        (4, ("no_prefix",), True),
        (4, ("uflpa", "psa_ofac_50_percent_rule"), True),
    ],
)
def test_psa_uses_source_count_and_published_type_only(
    count: int, factors: tuple[str, ...], expected: bool | None
) -> None:
    # The PSA count is copied as published, and qualification uses only the published risk type.
    item = profile(count=count, factors=factors)
    item.risk_factors.append(
        RiskFactor(
            factor="ordinary_factor",
            value="psa_ofac_50_percent_rule",
            metadata={"psa_ofac_50_percent_rule": True},
            level="critical",
        )
    )
    result = psa_exposure([supplier()], {"S1": item})
    assert list(result.columns) == PSA_COLUMNS
    record = result.to_dict("records")[0]
    assert record["psa_count"] == count and type(record["psa_count"]) is int
    assert record["psa_risky"] is expected
    assert record["psa_status"] == "available"
    assert item.risk_factors[0].factor == (factors[0] if factors else "ordinary_factor")


@pytest.mark.parametrize("profile_id", ["OTHER", "s1", " S1"])
def test_psa_rejects_profile_evidence_for_a_different_entity(profile_id: str) -> None:
    # A profile for a different entity can never supply this row's PSA evidence.
    row = supplier("S1")
    item = profile(profile_id, 77, ("psa_ofac_50_percent_rule",))
    before_row = row.model_copy(deep=True)
    before_profile = item.model_copy(deep=True)
    with pytest.raises(ValueError, match="Profile ID does not match supplier ID"):
        psa_exposure([row], {"S1": item})
    assert row == before_row and item == before_profile


def test_psa_preserves_duplicate_supplier_rows_nullable_types_and_diagnostics() -> None:
    # Duplicate workbook rows each keep their own PSA record, and null values stay null.
    rows = [
        supplier("\uae30\uc544", 4),
        supplier("ZERO", 3),
        supplier("\uae30\uc544", 1, "another_portfolio"),
        supplier("\uae30\uc544", 2),
        supplier("MISSING", 5),
        supplier("\uae30\uc544", 6),
    ]
    rows[-1].profile_error = "Synthetic profile failure"
    rows[-1].profile_error_type = "Timeout"
    profiles = {
        "\uae30\uc544": profile("\uae30\uc544", 4, ("psa_ofac_50_percent_rule",)),
        "ZERO": profile("ZERO", 0),
    }
    before_rows = [row.model_copy(deep=True) for row in rows]
    before_profiles = {key: value.model_copy(deep=True) for key, value in profiles.items()}
    result = psa_exposure(rows, profiles)
    records = result.to_dict("records")
    assert [
        (r["portfolio"], r["row_number"], r["psa_count"], r["psa_risky"], r["psa_status"])
        for r in records
    ] == [
        ("another_portfolio", 1, 4, True, "available"),
        ("list_3", 2, 4, True, "available"),
        ("list_3", 3, 0, False, "available"),
        ("list_3", 4, 4, True, "available"),
        ("list_3", 5, None, None, "unavailable"),
        ("list_3", 6, None, None, "unavailable"),
    ]
    assert records[0]["entity_id"] == "\uae30\uc544"
    assert all(dtype == "object" for dtype in result.dtypes)
    assert type(result.iloc[0]["psa_count"]) is int
    assert result.iloc[0]["psa_risky"] is True
    assert result.iloc[-1]["psa_count"] is None
    assert json.loads(json.dumps(records, allow_nan=False)) == records
    assert (
        json.loads(Findings(generated_at="synthetic", suppliers=records).model_dump_json())[
            "suppliers"
        ]
        == records
    )
    assert psa_exposure(list(reversed(rows)), profiles).to_dict("records") == records
    assert rows == before_rows and profiles == before_profiles


@pytest.mark.parametrize("entity_id", [None, "", " \t"])
def test_blank_resolved_id_is_unavailable_with_null_identity(entity_id: str | None) -> None:
    # A blank accepted ID makes the PSA evidence unavailable; it never reads as zero PSA exposure.
    record = psa_exposure([supplier(entity_id)], {"S1": profile()}).to_dict("records")[0]
    assert record["entity_id"] is None
    assert record["psa_status"] == "unavailable"
    assert record["psa_count"] is None and record["psa_risky"] is None


@pytest.mark.parametrize("status", ["weak", "no_match", "error"])
def test_psa_excludes_nonresolved_rows_even_with_a_cached_profile(status: str) -> None:
    # A cached profile cannot turn a match that was not accepted into PSA evidence.
    row = supplier().model_copy(update={"status": status})
    assert psa_exposure([row], {"S1": profile()}).empty


def test_empty_psa_has_declared_columns_and_ignores_orphan_profiles() -> None:
    # Empty PSA output keeps its declared columns, and profiles with no matching row are ignored.
    result = psa_exposure([], {"S1": profile()})
    assert result.empty and list(result.columns) == PSA_COLUMNS
    assert json.dumps(result.to_dict("records"), allow_nan=False) == "[]"

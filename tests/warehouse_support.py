"""Domain-model builders shared by the warehouse and traversal-selection tests."""

from sayari_poc.models import (
    EntityProfile,
    ResolvedEntity,
    RiskFactor,
    UpstreamEntity,
    UpstreamResult,
)


def supplier(entity_id: str = "S1", row: int = 2, sheet: str = "list_3") -> ResolvedEntity:
    return ResolvedEntity(
        row_number=row,
        sheet=sheet,
        input_name="Supplier " + entity_id,
        entity_id=entity_id,
        label="Resolved " + entity_id,
        translated_label="Translated " + entity_id,
        match_strength="strong",
        status="resolved",
    )


def node(entity_id: str = "U1") -> UpstreamEntity:
    return UpstreamEntity(
        entity_id=entity_id,
        label="\u77f3\u5bb6\u5e84\u6cf0\u660e\u987f",
        translated_label="Shijiazhuang Taimington",
        countries=["CHN", "MYS"],
        country_count=2,
        risk_factors=["owned_by_military_civil_fusion", "unknown_factor"],
    )


def result(entity_id: str, *entities: UpstreamEntity) -> UpstreamResult:
    return UpstreamResult(
        supplier_id=entity_id,
        entities={item.entity_id: item for item in entities},
        status="assessed" if entities else "no_data",
        partial_results=False,
        explored_count=42,
    )


def profile(entity_id: str = "S1") -> EntityProfile:
    return EntityProfile(
        entity_id=entity_id,
        label="Profile " + entity_id,
        translated_label=None,
        countries=["USA"],
        degree=10,
        psa_count=0,
        max_level="high",
        risk_factors=[RiskFactor(factor="raw_profile_factor", level="high", metadata={})],
    )

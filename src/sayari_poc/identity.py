"""Identity findings from existing profile evidence, without API access."""

import pandas as pd  # type: ignore[import-untyped]

from sayari_poc.models import EntityProfile, ResolvedEntity
from sayari_poc.risk_taxonomy import RiskOntology, load_ontology


def _object_frame(columns: list[str], rows: list[tuple[object, ...]]) -> pd.DataFrame:
    """Keep nullable integers out of floating-point coercion."""
    return pd.DataFrame(
        {
            column: pd.Series([row[index] for row in rows], dtype=object)
            for index, column in enumerate(columns)
        }
    )


def psa_exposure(
    suppliers: list[ResolvedEntity],
    profiles: dict[str, EntityProfile],
    ontology: RiskOntology | None = None,
) -> pd.DataFrame:
    """Preserve one PSA observation per resolved supplier input row.

    Profiles are the successful, retained evidence from fetch_profiles. A missing profile or
    row-level profile error means unavailable, never a measured zero. The flag uses published risk
    types; unresolved factors leave it unknown unless a published PSA factor is present. PSA count
    is copied without interpretation.
    """
    ontology = load_ontology() if ontology is None else ontology
    rows: list[tuple[object, ...]] = []
    for row in sorted(suppliers, key=lambda row: (row.sheet, row.row_number)):
        if row.status != "resolved":
            continue
        entity_id = row.entity_id if row.entity_id and row.entity_id.strip() else None
        profile = (
            profiles.get(entity_id) if entity_id is not None and row.profile_error is None else None
        )
        # Catch a profile filed under the wrong entity before we attach its evidence.
        if profile is not None and profile.entity_id != entity_id:
            raise ValueError(f"Profile ID does not match supplier ID: {entity_id!r}")
        published = (
            [ontology.factors.get(factor.factor) for factor in profile.risk_factors]
            if profile is not None
            else []
        )
        has_psa = any(item is not None and item.risk_type == "psa" for item in published)
        # One known PSA factor is enough for True, even if other factors are unknown. We only return
        # False when every observed factor is classified; otherwise the flag is None.
        rows.append(
            (
                row.sheet,
                row.row_number,
                entity_id,
                profile.psa_count if profile is not None else None,
                has_psa if profile is not None and (has_psa or all(published)) else None,
                "available" if profile is not None else "unavailable",
            )
        )
    return _object_frame(
        ["portfolio", "row_number", "entity_id", "psa_count", "psa_risky", "psa_status"], rows
    )

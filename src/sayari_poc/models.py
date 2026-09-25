"""Validated records passed between pipeline stages."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict  # Required by Pydantic on Python 3.11.


class InputEntity(BaseModel):
    """A named workbook row; row numbers refer to the original Excel sheet."""

    row_number: int = Field(gt=0)
    name: str = Field(min_length=1)
    address: str | None
    country: str | None
    sheet: str


class IngestDiagnostic(TypedDict):
    """Distinguish dropped rows from warnings about retained input entities."""

    row_number: str
    sheet: str
    reason: str
    # "dropped" rows are gone; "annotated" rows are kept and must not add to the row totals.
    kind: Literal["dropped", "annotated"]


class IngestResult(BaseModel):
    """Retained entities and separately counted ingestion diagnostics."""

    entities: list[InputEntity]
    exceptions: list[IngestDiagnostic]


class ResolutionMatchStrength(BaseModel):
    """Sayari's strong-or-weak category, independent of relevance."""

    value: Literal["strong", "weak"]


class ResolutionCandidate(BaseModel):
    """Candidate identity evidence in received response order."""

    model_config = ConfigDict(strict=True)

    entity_id: str = Field(pattern=r"\S")
    label: str = Field(pattern=r"\S")
    translated_label: str | None = None
    match_strength: ResolutionMatchStrength
    # Relevance is separate from the categorical match strength; never scale it to a percentage.
    # Verified in sayari/resolution/types/resolution_result.py: ResolutionResult.
    score: float | None = Field(default=None, allow_inf_nan=False)
    # None means an older projection; [] means Sayari returned no countries for this candidate.
    countries: list[str] | None = None


class ResolvedEntity(BaseModel):
    """One result per input row, with validated candidate evidence.

    candidate_count counts entries returned by the API, including malformed ones, not all possible
    matches or an ambiguity score. candidate_errors uses original zero-based response indices;
    candidates contains only validated entries. status/error/error_type describe resolution;
    profile_error/profile_error_type describe the latest profile attempt without changing that
    resolution outcome.
    """

    row_number: int = Field(gt=0)
    sheet: str
    input_name: str
    # The primary candidate's ID. Whether it counts as a match still depends on the row status.
    entity_id: str | None = None
    label: str | None = None
    translated_label: str | None = None
    match_strength: Literal["strong", "weak"] | None = None
    # The primary candidate's relevance score, kept for review. It is not a confidence value.
    score: float | None = None
    candidate_count: int = Field(default=0, ge=0)
    candidates: list[ResolutionCandidate] = Field(default_factory=list)
    candidate_errors: dict[int, str] = Field(default_factory=dict)
    status: Literal["resolved", "weak", "no_match", "error"]
    error: str | None = None
    error_type: str | None = None
    profile_error: str | None = None
    profile_error_type: str | None = None


class ProfileRiskValue(BaseModel):
    """The small part of a Sayari risk value needed by this PoC."""

    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    # Keep bool, int, float, str and missing values distinct; don't let one coerce into another.
    value: bool | int | float | str | None = None
    # The level as the profile reported it, kept apart from any later ontology classification.
    level: str | None = None
    metadata: dict[str, object]


class RiskFactor(ProfileRiskValue):
    """One named risk factor, with Sayari's severity and original metadata."""

    factor: str = Field(pattern=r"\S")


class EntityProfileResponse(BaseModel):
    """The consumed subset of a profile response."""

    model_config = ConfigDict(strict=True)

    id: str = Field(pattern=r"\S")
    label: str | None = None
    translated_label: str | None = None
    countries: list[str]
    psa_count: int = Field(ge=0)
    degree: int | None = Field(default=None, ge=0)
    risk: dict[str, ProfileRiskValue] = Field(default_factory=dict)


class EntityProfile(BaseModel):
    """Validated profile evidence, including nullable severity."""

    model_config = ConfigDict(strict=True)

    entity_id: str = Field(pattern=r"\S")
    label: str | None
    translated_label: str | None
    countries: list[str]
    # A non-negative PSA count. It does not confirm that the identities actually overlap.
    psa_count: int = Field(ge=0)
    degree: int | None = Field(ge=0)
    risk_factors: list[RiskFactor]
    # The highest recognised level in this profile. It says nothing about which factors are absent.
    max_level: str | None


class UpstreamEntity(BaseModel):
    """Canonical upstream identity and bounded retrieval evidence."""

    model_config = ConfigDict(strict=True)

    entity_id: str = Field(pattern=r"\S")
    label: str | None = None
    translated_label: str | None = None
    countries: list[str]
    country_count: int = Field(ge=0)
    risk_factors: list[str]


class PathComponent(BaseModel):
    """One trade-record evidence bundle attached to a hop."""

    model_config = ConfigDict(strict=True)

    hs_code: str | None = None
    departure_countries: list[str]
    arrival_countries: list[str]
    # The earliest date a trade record was observed. It is not a contract start date.
    min_date: str | None = None
    # The latest date a trade record was observed. It is not a contract end date.
    max_date: str | None = None


class SupplyPathHop(BaseModel):
    """One Sayari-annotated tier/entity in source order."""

    model_config = ConfigDict(strict=True)

    # The tier exactly as Sayari annotated it; we don't recompute graph distance. Verified in
    # sayari/supply_chain/types/trade_traversal_path_segment.py: TradeTraversalPathSegment.
    tier: int
    entity_id: str = Field(pattern=r"\S")
    components: list[PathComponent]


class SupplyPath(BaseModel):
    """A retrieved trade path with its original response index."""

    model_config = ConfigDict(strict=True)

    source_entity_id: str = Field(pattern=r"\S")
    path_index: int = Field(ge=0)
    hops: list[SupplyPathHop] = Field(min_length=1)


class UpstreamResult(BaseModel):
    """Bounded retrieval evidence or an explicit failure outcome.

    no_data is absence of upstream evidence, not absence of risk. On errors, metadata retains valid
    received values; False/None otherwise are placeholders and cannot establish coverage.
    explored_count is never an entity count.
    """

    model_config = ConfigDict(strict=True)

    supplier_id: str
    entities: dict[str, UpstreamEntity]
    # Sayari's partial-results flag. On an error this is only a placeholder, not a real reading.
    partial_results: bool
    # Sayari's exploration diagnostic. It is not the number of entities we retrieved.
    explored_count: int | None
    status: Literal["assessed", "partial", "no_data", "error"]
    error_type: str | None = None
    paths: list[SupplyPath] = Field(default_factory=list)


class OntologyFactor(BaseModel):
    """Published ontology fields; absent factors are represented separately."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str
    categories: list[str]
    level: Literal["critical", "high", "elevated", "relevant"]
    risk_type: Literal["seed", "network", "psa"]


class Findings(BaseModel):
    """Complete stage evidence that the static report is rendered from."""

    # When the artifacts were generated (declared or carried over). It says nothing about how fresh
    # the source data is.
    generated_at: str
    suppliers: list[dict[str, object]]
    shared_nodes: list[dict[str, object]] = Field(default_factory=list)
    convergence: dict[str, dict[str, object]] = Field(default_factory=dict)
    coverage: dict[str, dict[str, int]] = Field(default_factory=dict)
    exceptions: list[dict[str, str]] = Field(default_factory=list)
    # Deterministic run parameters and evidence provenance; per-run counters are left out.
    manifest: dict[str, object] = Field(default_factory=dict)
    paths: dict[str, object] = Field(default_factory=dict)
    # The published definitions of the factors we saw, embedded so rendering needs no ontology I/O.
    ontology: dict[str, OntologyFactor] = Field(default_factory=dict)

"""Assemble completed stage evidence for deterministic report artifacts."""

import json
from collections.abc import Mapping, Sequence
from typing import cast

from sayari_poc.config import Settings
from sayari_poc.models import (
    EntityProfile,
    Findings,
    IngestResult,
    InputEntity,
    ResolvedEntity,
    SupplyPath,
    UpstreamResult,
)
from sayari_poc.risk_taxonomy import RiskOntology, load_ontology

# Cap the path examples stored per ranked node; the full retrieved count is still reported.
PATH_MAX_PER_NODE = 6


def _path_sort_key(path: SupplyPath) -> tuple[tuple[tuple[int, str], ...], str, int]:
    """Order retained paths without inferring tier annotations."""
    return (
        tuple((hop.tier, hop.entity_id) for hop in path.hops),
        json.dumps(
            [[component.model_dump() for component in hop.components] for hop in path.hops],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        path.path_index,
    )


def _path_evidence(
    upstream: Mapping[str, UpstreamResult],
    ranked: list[dict[str, object]],
    headline_portfolio: str | None,
) -> dict[str, object]:
    """Balance capped path examples and disclose retrieved totals.

    Selection is round-robin in canonical supplier order. Hop identities are embedded so the
    report needs no further retrieval.
    """
    entities: dict[str, dict[str, str | None]] = {}
    nodes: dict[str, object] = {}
    if headline_portfolio is None:
        return {"entities": entities, "nodes": nodes}
    for node in ranked:
        if node["portfolio"] != headline_portfolio:
            continue
        node_id = cast(str, node["upstream_id"])
        groups: dict[str, list[SupplyPath]] = {}
        for supplier in cast(list[dict[str, object]], node["suppliers"]):
            retrieval = upstream.get(cast(str, supplier["entity_id"]))
            # A missing retrieval means we have no evidence, not that there is no route.
            if retrieval is None:
                continue
            for path in retrieval.paths:
                # A path is evidence for every entity it passes through, not just its last hop.
                if any(hop.entity_id == node_id for hop in path.hops):
                    groups.setdefault(path.source_entity_id, []).append(path)
        for group in groups.values():
            group.sort(key=_path_sort_key)
        sources = sorted(groups)
        retrieved_count = sum(len(group) for group in groups.values())
        selected: list[SupplyPath] = []
        round_index = 0
        # Hand out the PATH_MAX_PER_NODE slots one supplier at a time, so a supplier with many paths
        # can't crowd out the others.
        while len(selected) < min(PATH_MAX_PER_NODE, retrieved_count):
            for source in sources:
                if round_index < len(groups[source]):
                    selected.append(groups[source][round_index])
                    if len(selected) == PATH_MAX_PER_NODE:
                        break
            round_index += 1
        for path in selected:
            for hop in path.hops:
                entity = upstream[path.source_entity_id].entities[hop.entity_id]
                entities.setdefault(
                    hop.entity_id,
                    {"label": entity.label, "translated_label": entity.translated_label},
                )
        nodes[node_id] = {
            "retrieved_path_count": retrieved_count,
            "stored_path_count": len(selected),
            "truncated": len(selected) < retrieved_count,
            "path_observed_supplier_ids": sources,
            "paths": [path.model_dump(mode="json") for path in selected],
        }
    return {"entities": dict(sorted(entities.items())), "nodes": nodes}


def _row_exceptions(row: ResolvedEntity) -> list[dict[str, str]]:
    """Keep each row's stage and candidate diagnostics distinct."""
    identity = {"sheet": row.sheet, "row_number": str(row.row_number), "input_name": row.input_name}
    diagnostics: list[dict[str, str]] = []
    if row.status in {"weak", "no_match", "error"}:
        reason = row.error or (
            "Weak match: needs adjudication; profile not requested"
            if row.status == "weak"
            else "Resolution failed; profile not requested"
            if row.status == "error"
            else "No matching entity; profile not requested"
        )
        diagnostics.append(
            {
                **identity,
                "stage": "resolution",
                "reason": reason,
                "error_type": row.error_type or row.status,
            }
        )
    # Sort malformed-candidate diagnostics by their numeric response position, so 10 comes after 2.
    for index, reason in sorted(row.candidate_errors.items()):
        diagnostics.append(
            {
                **identity,
                "stage": "resolution",
                "reason": f"Candidate {index}: {reason}",
                "error_type": "ValidationError",
            }
        )
    if row.profile_error:
        diagnostics.append(
            {
                **identity,
                "stage": "profile",
                "reason": row.profile_error,
                "error_type": row.profile_error_type or "ProfileError",
            }
        )
    return diagnostics


def assemble_findings(
    *,
    generated_at: str,
    ingested: Mapping[str, IngestResult],
    inputs: Sequence[InputEntity],
    resolved: Sequence[ResolvedEntity],
    profiles: Mapping[str, EntityProfile],
    upstream: Mapping[str, UpstreamResult],
    psa_records: Sequence[Mapping[str, object]],
    ranked: list[dict[str, object]],
    convergence: dict[str, dict[str, object]],
    coverage: dict[str, dict[str, int]],
    headline_portfolio: str | None,
    limit: int | None,
    settings: Settings,
    ontology: RiskOntology | None = None,
) -> Findings:
    """Assemble stable evidence and diagnostics for every input row."""
    ontology = load_ontology() if ontology is None else ontology
    observed = sorted(
        {factor.factor for profile in profiles.values() for factor in profile.risk_factors}
        | {
            factor
            for result in upstream.values()
            for entity in result.entities.values()
            for factor in entity.risk_factors
        }
    )
    published = {
        factor: ontology.factors[factor] for factor in observed if factor in ontology.factors
    }
    unresolved = [factor for factor in observed if factor not in published]
    # Key PSA observations by workbook row so duplicate canonical suppliers stay separate rows.
    psa = {(row["portfolio"], row["row_number"]): row for row in psa_records}
    suppliers: list[dict[str, object]] = []
    exceptions: list[dict[str, str]] = [
        {
            "sheet": diagnostic["sheet"],
            "row_number": diagnostic["row_number"],
            "kind": diagnostic["kind"],
            "reason": diagnostic["reason"],
            "stage": "ingestion",
        }
        for result in ingested.values()
        for diagnostic in result.exceptions
    ]
    # resolve_entities keeps input order; the strict zip checks that lengths match, not rows.
    for source, row in zip(inputs, resolved, strict=True):
        profile = (
            profiles.get(row.entity_id or "")
            if row.status == "resolved" and not row.profile_error
            else None
        )
        supplier: dict[str, object] = row.model_dump(mode="json")
        supplier.update(
            {
                "portfolio": row.sheet,
                "input_address": source.address,
                "input_country": source.country,
                "resolution_status": row.status,
                "status": "error" if row.profile_error else row.status,
                "profile_status": "error"
                if row.profile_error
                else ("available" if profile is not None else "not_requested"),
                "profile": profile.model_dump(mode="json") if profile is not None else None,
            }
        )
        # Attach upstream results to resolved rows only, even when a weak candidate shares the ID.
        retrieval = upstream.get(row.entity_id or "") if (row.status == "resolved") else None
        supplier.update(
            {
                "coverage_status": retrieval.status if retrieval is not None else None,
                "upstream_entity_count": (
                    sum(entity_id != retrieval.supplier_id for entity_id in retrieval.entities)
                    if retrieval is not None and retrieval.status != "error"
                    else None
                ),
                "upstream_partial_results": (
                    retrieval.partial_results
                    if retrieval is not None and retrieval.status != "error"
                    else None
                ),
                "upstream_error_type": retrieval.error_type if retrieval is not None else None,
                "psa_count": None,
                "psa_risky": None,
                "psa_status": "not_resolved",
            }
        )
        observation = psa.pop((row.sheet, row.row_number), None)
        if observation is not None:
            if observation["entity_id"] != row.entity_id:
                raise ValueError("PSA entity identity differs from supplier row")
            supplier.update(
                {key: observation[key] for key in ("psa_count", "psa_risky", "psa_status")}
            )
        # Record a failed upstream retrieval as an exception, not as a successful empty result.
        if retrieval is not None and retrieval.status == "error":
            exceptions.append(
                {
                    "sheet": row.sheet,
                    "row_number": str(row.row_number),
                    "input_name": row.input_name,
                    "stage": "upstream",
                    "reason": "Upstream retrieval failed; bounded evidence unavailable",
                    "error_type": retrieval.error_type or "UpstreamError",
                }
            )
        suppliers.append(supplier)
        exceptions.extend(_row_exceptions(row))
    if psa:
        # Fail on PSA evidence with no supplier row rather than silently dropping it.
        raise ValueError("PSA evidence has no matching supplier row")
    suppliers.sort(key=lambda row: (str(row["portfolio"]), int(str(row["row_number"]))))
    # Publish evidence, bounds, ontology provenance and path examples; no per-run counters.
    return Findings(
        generated_at=generated_at,
        suppliers=suppliers,
        shared_nodes=ranked,
        convergence=convergence,
        coverage=coverage,
        exceptions=exceptions,
        manifest={
            "sheets": list(ingested),
            "limit": limit,
            "input_entities": len(inputs),
            "headline_portfolio": headline_portfolio,
            "hub_max_countries": settings.hub_max_countries,
            "path_max_per_node": PATH_MAX_PER_NODE,
            "max_upstream_depth": settings.max_upstream_depth,
            "upstream_limit": settings.upstream_limit,
            "taxonomy": {
                **ontology.parameters(),
                "observed_factor_count": len(observed),
                "resolved_factor_count": len(published),
                "unresolved_factor_count": len(unresolved),
                "unresolved_factors": unresolved,
                "missing_definition_count": sum(
                    not item.description.strip() for item in published.values()
                )
                + len(unresolved),
            },
        },
        paths=_path_evidence(upstream, ranked, headline_portfolio),
        ontology=published,
    )

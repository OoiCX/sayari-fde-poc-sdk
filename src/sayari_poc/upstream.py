"""Retrieve bounded upstream trade evidence only for resolved identities."""

import logging

from sayari_poc.config import Settings
from sayari_poc.models import ResolvedEntity, UpstreamEntity, UpstreamResult
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import CallState, SayariError, SayariValidationError

_LOGGER = logging.getLogger(__name__)


def fetch_upstream(
    client: SayariClient,
    supplier_id: str,
    settings: Settings,
    *,
    observed_entities: dict[str, UpstreamEntity] | None = None,
) -> UpstreamResult:
    """Retrieve bounded evidence with isolated supplier failures.

    Coverage applies only within the requested bounds; retrieved counts are lower bounds. On
    error, False/None metadata can be placeholders and must never be interpreted as measured
    coverage. Valid received metadata remains diagnostic.
    """
    try:
        result = client.upstream(
            supplier_id, max_depth=settings.max_upstream_depth, limit=settings.upstream_limit
        )
        if result.status == "error":
            return result
        state = CallState(
            partial_results=result.partial_results, explored_count=result.explored_count
        )
        if result.supplier_id != supplier_id or (
            result.status == "no_data" and (result.entities or result.paths)
        ):
            raise SayariValidationError("Inconsistent upstream result", call_state=state)
        # Collect normalised entities here first, so a half-validated supplier is never shared.
        normalized: dict[str, UpstreamEntity] = {}
        for entity_id, entity in sorted(result.entities.items()):
            if (
                entity_id != entity.entity_id
                or entity.country_count != len(entity.countries)
                or any(not factor.strip() for factor in entity.risk_factors)
            ):
                raise SayariValidationError("Malformed upstream entity", call_state=state)
            # Sort countries and factor IDs in a copy used only for comparison; the returned
            # evidence is left as received.
            normalized[entity_id] = entity.model_copy(
                update={
                    "countries": sorted(entity.countries),
                    "risk_factors": sorted(entity.risk_factors),
                }
            )
            # Fail if an earlier, fully validated supplier saw this same entity differently.
            if (
                observed_entities is not None
                and entity_id in observed_entities
                and observed_entities[entity_id] != normalized[entity_id]
            ):
                raise SayariValidationError("Conflicting upstream observation", call_state=state)
        # Every path must belong to this supplier, and every hop must point at a retrieved entity.
        if any(
            path.source_entity_id != supplier_id
            or any(hop.entity_id not in result.entities for hop in path.hops)
            for path in result.paths
        ):
            raise SayariValidationError("Malformed upstream path identity", call_state=state)
        # Only share fully validated suppliers; a rejected one mustn't skew later conflict checks.
        if observed_entities is not None:
            observed_entities.update(normalized)
        return result
    except Exception as exc:
        result = UpstreamResult(
            supplier_id=supplier_id,
            entities={},
            partial_results=False,
            explored_count=None,
            status="error",
            error_type=(
                "ValidationError" if isinstance(exc, SayariValidationError) else type(exc).__name__
            ),
        )
        if isinstance(exc, SayariError):
            # Keep a valid partial-results flag as a diagnostic. It is not measured coverage.
            result.partial_results = exc.call_state.partial_results
            # Keep a valid exploration count as a diagnostic. It is not an entity total.
            result.explored_count = exc.call_state.explored_count
        _LOGGER.error(
            "Upstream retrieval failed for supplier %r (%s)", supplier_id, result.error_type
        )
        return result


def fetch_upstreams(
    client: SayariClient,
    resolved: list[ResolvedEntity],
    settings: Settings,
) -> dict[str, UpstreamResult]:
    """Retrieve each distinct resolved ID once, in sorted order.

    Rows retain their resolution evidence. Weak, no-match and failed rows are unattempted, so they
    have no upstream result, including when a weak candidate shares an ID with a resolved row.
    Consumers must check row resolution status.
    """
    # Take only the canonical IDs of resolved rows, sorted; a weak candidate never triggers a fetch.
    supplier_ids = sorted(
        {
            row.entity_id
            for row in resolved
            if row.status == "resolved" and row.entity_id is not None
        }
    )
    observed_entities: dict[str, UpstreamEntity] = {}
    return {
        entity_id: fetch_upstream(client, entity_id, settings, observed_entities=observed_entities)
        for entity_id in supplier_ids
    }

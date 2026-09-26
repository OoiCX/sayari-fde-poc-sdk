"""Rank screened supplier rows for triage using bounded evidence.

Keep one row per workbook row, even when several resolve to the same canonical identity.
Coverage accompanies every count, and both assessed and partial retrieval remain bounded. Counts
are lower bounds, not scores.
"""

from collections import Counter
from typing import Any

from sayari_poc.models import EntityProfile, ResolvedEntity, UpstreamResult
from sayari_poc.risk_taxonomy import SELECTED_LEVELS, RiskOntology

NOT_ATTEMPTED = "not_attempted"


def _share(count: int, largest: int) -> int:
    """Return a rounded share of the busiest row, or zero if empty.

    The bar compares rows against each other, not against any population, so it carries no
    absolute meaning and the report says so.
    """
    return round(100 * count / largest) if largest else 0


def supplier_breakdown(
    portfolio: str,
    resolved: list[ResolvedEntity],
    profiles: dict[str, EntityProfile],
    upstream: dict[str, UpstreamResult],
    nodes: list[dict[str, Any]],
    ontology: RiskOntology,
) -> list[dict[str, object]]:
    """Return one row per screened supplier row, most connected first.

    A supplier's flagged count is how many of this list's shared entities it reaches in
    retrieved evidence. It is not a count of that supplier's own findings, and it is not an
    assertion that the supplier is at fault.
    """
    reach: Counter[str] = Counter()
    for node in nodes:
        reach.update(
            {
                str(supplier["entity_id"])
                for supplier in node.get("suppliers") or ()
                if supplier.get("entity_id")
            }
        )
    rows: list[dict[str, object]] = []
    for row in resolved:
        if row.sheet != portfolio:
            continue
        # Only a resolved row may use canonical evidence it shares with another row.
        entity_id = row.entity_id if row.status == "resolved" else None
        # Skip a profile this row failed to fetch, even if another row with this identity has one.
        profile = profiles.get(entity_id) if entity_id and not row.profile_error else None
        retrieval = upstream.get(entity_id) if entity_id else None
        # A failed or unattempted search gives no measured upstream count.
        retrieved = retrieval is not None and retrieval.status != "error"
        # Count this supplier's own profile factors using published ontology levels. These worklist
        # columns do not select shared entities; that rule uses upstream factors only.
        levels: Counter[str] = Counter(
            ontology.factors[observed.factor].level
            for observed in (profile.risk_factors if profile else ())
            if observed.factor in ontology.factors
        )
        rows.append(
            {
                "entity_id": entity_id,
                "label": (row.label or row.input_name) if entity_id else row.input_name,
                # A row we never searched is a different state from one that returned nothing.
                "coverage": retrieval.status if retrieval is not None else NOT_ATTEMPTED,
                # The traversal root stays in the raw evidence but isn't upstream of itself.
                "upstream": sum(key != entity_id for key in retrieval.entities)
                if retrieved and retrieval is not None
                else None,
                "flagged": reach.get(entity_id, 0) if retrieved and entity_id else None,
                "severe_factors": sum(levels[level] for level in SELECTED_LEVELS)
                if profile is not None
                else None,
                # Sayari documents levels as ordered severity with no per-level action, and the
                # selection rule treats critical and high as one set, so the report shows one
                # combined count. Carry the critical part separately anyway, so that if a later
                # snapshot surfaces one it doesn't vanish inside that total.
                "critical_factors": levels["critical"] if profile is not None else None,
                # Elevated factors don't qualify an entity, but leaving them out could make a row
                # with no selected factors look as if it had no observed risk.
                "elevated_factors": levels["elevated"] if profile is not None else None,
            }
        )
    largest = max((int(str(record["flagged"] or 0)) for record in rows), default=0)
    for record in rows:
        record["flagged_share"] = _share(int(str(record["flagged"] or 0)), largest)
    # Order by reach, then published severity, then retrieved breadth, then name and ID, so ties
    # never reorder between runs and the artifacts stay byte-identical.
    rows.sort(
        key=lambda record: (
            -int(str(record["flagged"] or 0)),
            -int(str(record["severe_factors"] or 0)),
            -int(str(record["upstream"] or 0)),
            str(record["label"]),
            str(record["entity_id"]),
        )
    )
    return rows

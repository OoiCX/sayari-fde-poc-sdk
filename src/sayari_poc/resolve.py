"""Resolve workbook identities and retain uncertain matches for adjudication."""

import logging
from traceback import walk_tb

from sayari_poc.models import InputEntity, ResolvedEntity
from sayari_poc.sayari_sdk import SayariClient

_LOGGER = logging.getLogger(__name__)


def resolve_entities(client: SayariClient, entities: list[InputEntity]) -> list[ResolvedEntity]:
    """Resolve every input in order, retaining failure outcomes.

    The facade preserves Sayari's candidate order and raw relevance scores. Only an explicitly
    strong primary match permits canonical acceptance.
    """
    results: list[ResolvedEntity] = []
    for entity in entities:
        try:
            result = client.resolve(entity)
        except Exception as exc:
            # Keep the failed row's identity with a fixed message; don't copy the exception text.
            result = ResolvedEntity(
                row_number=entity.row_number,
                sheet=entity.sheet,
                input_name=entity.name,
                status="error",
                error="Unexpected resolution failure",
                error_type=type(exc).__name__,
            )
            # For debugging, log only function names and line numbers.
            locations = [(frame.f_code.co_name, line) for frame, line in walk_tb(exc.__traceback__)]
            _LOGGER.error(
                "Resolution failed for row %s (%s), stack locations: %s",
                entity.row_number,
                result.error_type,
                locations,
            )
        results.append(result)
    return results

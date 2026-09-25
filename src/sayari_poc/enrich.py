"""Retrieve resolved entity profiles and preserve Sayari's risk evidence."""

import logging
from traceback import walk_tb

from pydantic import ValidationError

from sayari_poc.models import EntityProfile, ResolvedEntity
from sayari_poc.sayari_sdk import SayariClient
from sayari_poc.transport import SayariError, SayariValidationError

_LOGGER = logging.getLogger(__name__)


def fetch_profiles(
    client: SayariClient, resolved: list[ResolvedEntity]
) -> dict[str, EntityProfile]:
    """Fetch accepted profiles and record failures on their input rows.

    The caller selects the supplier portfolio. Only resolved rows are fetched; resolution status
    and evidence stay intact. Failures set profile_error/profile_error_type in place, allowing a
    later call to retry. A later success for the same ID clears earlier profile failures. Later
    cache hits can succeed after BudgetExceeded; the client owns the ceiling.
    """
    profiles: dict[str, EntityProfile] = {}
    for row in resolved:
        if row.status != "resolved":
            continue
        row.profile_error = row.profile_error_type = None
        entity_id = row.entity_id
        if entity_id is None or not entity_id.strip():
            row.profile_error = "Resolved entity is missing a usable entity ID"
            row.profile_error_type = "ValidationError"
            continue
        if entity_id in profiles:
            continue
        try:
            profiles[entity_id] = client.get_entity(entity_id)
        except (ValidationError, SayariValidationError):
            # Pydantic's error text includes the raw input, so record a fixed message instead.
            row.profile_error = "Malformed entity profile"
            row.profile_error_type = "ValidationError"
        except SayariError as exc:
            row.profile_error = "Entity profile retrieval failed"
            row.profile_error_type = type(exc).__name__
        except Exception as exc:
            row.profile_error = "Unexpected entity profile failure"
            row.profile_error_type = type(exc).__name__
            locations = [(frame.f_code.co_name, line) for frame, line in walk_tb(exc.__traceback__)]
            # Log the row, failure type and stack positions, but no source or exception text.
            _LOGGER.error(
                "Profile retrieval failed for row %s (%s), stack locations: %s",
                row.row_number,
                row.profile_error_type,
                locations,
            )
    # If a duplicate row failed earlier but the same ID succeeded later, clear the stale error.
    for row in resolved:
        if row.status == "resolved" and row.entity_id in profiles:
            row.profile_error = row.profile_error_type = None
    return profiles

"""Load pinned ontology evidence without inferring unknown factor levels."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sayari_poc.models import OntologyFactor

# Classify against the dated snapshot committed alongside the project source.
SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2] / "data/public/sayari-risk-factors-2026-09-21.json"
)
SELECTED_LEVELS = ("critical", "high")


class OntologySnapshotError(ValueError):
    """Required ontology evidence is missing or invalid."""


@dataclass(frozen=True)
class RiskOntology:
    """Validated factor definitions and their capture provenance."""

    factors: dict[str, OntologyFactor]
    provenance: dict[str, Any]

    def parameters(self) -> dict[str, Any]:
        """Return independent provenance and selection-policy metadata."""
        selected = sum(1 for factor in self.factors.values() if factor.level in SELECTED_LEVELS)
        return {
            **self.provenance,
            "selected_levels": list(SELECTED_LEVELS),
            "selected_factor_count": selected,
        }


def load_ontology(path: Path | None = None) -> RiskOntology:
    """Verify and load the pinned, unfiltered ontology snapshot."""
    path = SNAPSHOT_PATH if path is None else path
    try:
        raw = path.read_bytes()
        meta = json.loads(path.with_suffix(".metadata.json").read_bytes())
        payload = json.loads(raw)
        captured = datetime.fromisoformat(meta["captured_at"])
        if (
            captured.tzinfo is None
            or meta["filters"] != {}
            or meta["sha256"] != hashlib.sha256(raw).hexdigest()
            or meta["sdk_version"] != "0.1.43"
            or any(value is not None for value in payload["filters"].values())
            or not payload["data"]
            or payload["total_count"] != len(payload["data"])
        ):
            raise ValueError
        factors = [OntologyFactor.model_validate(item) for item in payload["data"]]
        # Reject duplicate factor IDs; otherwise one would silently overwrite another in the lookup.
        if len({item.id for item in factors}) != len(factors):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Say how to restore the snapshot without exposing paths, payloads or the exception chain.
        raise OntologySnapshotError(
            "Required Sayari risk ontology snapshot is missing or invalid; "
            "restore the committed snapshot and metadata. No reclassification or network fallback."
        ) from None
    return RiskOntology(
        factors={item.id: item for item in sorted(factors, key=lambda item: item.id)},
        provenance={
            "snapshot": path.name,
            "captured_at": meta["captured_at"],
            "sha256": meta["sha256"],
            "filters": {},
            "sdk_version": meta["sdk_version"],
            "factor_count": len(factors),
        },
    )

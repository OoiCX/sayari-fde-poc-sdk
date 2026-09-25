"""Explicit source records for synthetic pipeline and classification evidence."""

import pytest

from sayari_poc.models import OntologyFactor
from sayari_poc.risk_taxonomy import RiskOntology


@pytest.fixture
def pipeline_ontology(monkeypatch: pytest.MonkeyPatch) -> RiskOntology:
    """Isolate synthetic pipeline contracts from the acquired source snapshot."""
    ontology = synthetic_ontology()
    monkeypatch.setattr("sayari_poc.pipeline.load_ontology", lambda: ontology)
    return ontology


def synthetic_ontology() -> RiskOntology:
    rows = [
        ("owned_by_military_civil_fusion", "high", "network"),
        ("owned_by_soe", "high", "network"),
        ("uflpa", "critical", "seed"),
        ("a_uflpa", "high", "seed"),
        ("b_uflpa", "high", "seed"),
        ("z_uflpa", "high", "seed"),
        ("export_to_soe", "relevant", "network"),
        ("exports_ilab_child_labor", "elevated", "network"),
        ("exports_ilab_forced_labor", "elevated", "network"),
        ("sanctioned_synthetic", "critical", "seed"),
        ("military_synthetic", "high", "network"),
        ("ilab_forced_labor", "elevated", "network"),
        ("psa_synthetic", "high", "psa"),
        ("psa_ofac_50_percent_rule", "high", "psa"),
        ("ordinary_factor", "relevant", "seed"),
        ("psa_misleading_name", "relevant", "network"),
        ("no_prefix", "high", "psa"),
    ]
    factors = {
        slug: OntologyFactor.model_validate(
            {
                "id": slug,
                "label": "Published " + slug.replace("_", " "),
                "description": "Synthetic published definition for " + slug.replace("_", " ") + ".",
                "categories": ["synthetic"],
                "level": level,
                "risk_type": kind,
            }
        )
        for slug, level, kind in rows
    }
    return RiskOntology(
        factors,
        {
            "snapshot": "synthetic",
            "captured_at": "2026-09-21T00:00:00+00:00",
            "sha256": "synthetic",
            "filters": {},
            "sdk_version": "0.1.43",
            "factor_count": len(factors),
        },
    )

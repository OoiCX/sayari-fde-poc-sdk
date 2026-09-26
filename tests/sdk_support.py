"""Synthetic SDK responses and a queued cache that drive the pipeline stages offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.sayari_sdk import SayariClient


def profile(entity_id: str = "root") -> dict[str, Any]:
    return {
        "id": entity_id,
        "label": "Synthetic profile",
        "degree": 0,
        "closed": False,
        "entity_url": "",
        "pep": False,
        "psa_count": 0,
        "sanctioned": False,
        "type": "company",
        "identifiers": [],
        "countries": ["SGP"],
        "source_count": {},
        "addresses": [],
        "trade_count": {},
        "relationship_count": {},
        "user_relationship_count": {},
        "attribute_count": {},
        "user_attribute_count": {},
        "related_entities_count": 0,
        "user_related_entities_count": 0,
        "user_record_count": 0,
        "risk": {},
    }


def upstream_payload(supplier_id: str = "root") -> dict[str, Any]:
    return {
        "filters": {},
        "partial_results": False,
        "explored_count": 1,
        "data": {
            "entities": {
                "node": {
                    "id": "node",
                    "type": "company",
                    "label": "Synthetic node",
                    "countries": ["USA", "SGP"],
                    "risk_factors": [],
                    "translated_label": "Synthetic translation",
                },
            },
            "paths": [
                {
                    "source_entity_id": supplier_id,
                    "path": [
                        {
                            "entity_id": "node",
                            "tier": 3,
                            "components": [
                                {
                                    "hs_code": "1234",
                                    "arrival_countries": ["SGP", "USA"],
                                    "departure_countries": ["CHN"],
                                    "min_date": "2020-01-01",
                                },
                            ],
                        },
                    ],
                }
            ],
        },
    }


def candidate(entity_id: str, strength: str = "strong", score: float = 130.5) -> dict[str, Any]:
    return {
        "profile": "corporate",
        "score": score,
        "entity_id": entity_id,
        "label": "Synthetic",
        "type": "company",
        "identifiers": [],
        "addresses": [],
        "countries": [],
        "sources": [],
        "typed_matched_queries": [],
        "matched_queries": [],
        "highlight": {},
        "explanation": {},
        "match_strength": {"value": strength},
    }


class QueuedCache(ResponseCache):
    def __init__(self, directory: Path, payloads: tuple[object, ...]) -> None:
        super().__init__(directory)
        self.payloads = iter(payloads)
        self.keys: list[str] = []

    def get(self, key: str) -> bytes | None:
        self.keys.append(key)
        payload = next(self.payloads, None)
        if isinstance(payload, BaseException):
            raise payload
        return json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None


def client_for(tmp_path: Path, *payloads: object) -> tuple[SayariClient, QueuedCache]:
    settings = Settings(_env_file=None, cache_dir=tmp_path / "cache")
    cache = QueuedCache(settings.cache_dir, payloads)
    return SayariClient(settings, cache, offline=True), cache


# The path's hop names an entity missing from the response, as conformance tests need.
def dangling_upstream_payload() -> dict[str, Any]:
    return {
        "filters": {},
        "partial_results": False,
        "explored_count": 1,
        "data": {
            "paths": [
                {
                    "source_entity_id": "root",
                    "path": [
                        {
                            "tier": 2,
                            "entity_id": "missing-node",
                            "components": [
                                {
                                    "hs_code": "1234",
                                    "arrival_countries": ["SGP"],
                                    "departure_countries": ["USA"],
                                }
                            ],
                        }
                    ],
                }
            ],
            "entities": {
                "root": {
                    "id": "root",
                    "type": "company",
                    "label": "Synthetic entity",
                    "risk_factors": [],
                    "countries": [],
                    "translated_label": "Synthetic translation",
                }
            },
        },
    }

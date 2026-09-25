"""Shared fixtures use synthetic credentials, temporary storage, and no network."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from openpyxl import Workbook
from pydantic import SecretStr

from sayari_poc import pipeline
from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.models import Findings
from sayari_poc.transport import AuditedTransport
from tests.ontology_support import synthetic_ontology
from tests.unit.test_p2_stages import candidate as sdk_candidate
from tests.unit.test_p2_stages import profile as sdk_profile


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return Settings(
        _env_file=None,
        sayari_client_id=f"synthetic-{uuid4()}",
        sayari_client_secret=SecretStr("synthetic-secret"),
        entity_file_path=tmp_path / "unused.xlsx",
        cache_dir=tmp_path / "cache",
    )


def resolution_payload(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Add SDK envelope fields only when constructing synthetic resolution evidence."""
    return {
        "fields": {},
        "data": [
            {**sdk_candidate(item["entity_id"]), **item} if "entity_id" in item else item
            for item in data
        ],
    }


def entity_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Complete a synthetic profile before tests apply any deliberate malformation."""
    return {**sdk_profile(data["id"]), **data}


@pytest.fixture
def cache(settings: Settings) -> ResponseCache:
    return ResponseCache(settings.cache_dir)


def _prepare(
    settings: Settings,
    tmp_path: Path,
    sheets: dict[str, list[tuple[str, str | None]]],
) -> tuple[Settings, ResponseCache]:
    configured = settings.model_copy(
        update={
            "entity_file_path": tmp_path / "input.xlsx",
            "duckdb_path": tmp_path / "analysis.duckdb",
            "hub_max_countries": 2,
        }
    )
    cache = ResponseCache(configured.cache_dir)
    workbook = Workbook()
    workbook.remove(cast(Any, workbook.active))
    for portfolio, rows in sheets.items():
        sheet = workbook.create_sheet(portfolio)
        sheet.append(["name", "address", "country"])
        for name, entity_id in rows:
            sheet.append([name, "Synthetic address", "SGP"])
            cache.put(
                cache.key(
                    httpx.Request(
                        "GET",
                        configured.sayari_api_base + "/v1/resolution",
                        params={
                            "name": name,
                            "address": "Synthetic address",
                            "country": "SGP",
                        },
                    ),
                ),
                json.dumps(
                    resolution_payload(
                        [
                            {
                                "entity_id": entity_id,
                                "label": f"供应商 {entity_id}",
                                "translated_label": f"Supplier {entity_id}",
                                "match_strength": {"value": "strong"},
                            }
                        ]
                        if entity_id
                        else []
                    ),
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            if entity_id:
                cache.put(
                    cache.key(
                        httpx.Request(
                            "GET",
                            configured.sayari_api_base + f"/v1/entity_summary/{entity_id}",
                            params={},
                        )
                    ),
                    json.dumps(
                        entity_payload(
                            {
                                "id": entity_id,
                                "label": f"供应商 {entity_id}",
                                "translated_label": f"Supplier {entity_id}",
                                "countries": ["SGP"],
                                "psa_count": 2,
                                "degree": 8,
                                "risk": {
                                    "psa_synthetic": {
                                        "value": True,
                                        "metadata": {},
                                        "level": "high",
                                    }
                                },
                            }
                        ),
                        ensure_ascii=False,
                    ).encode("utf-8"),
                )
                _upstream(cache, configured, entity_id, {})
    workbook.save(configured.entity_file_path)
    workbook.close()
    return configured, cache


def _upstream(
    cache: ResponseCache,
    settings: Settings,
    entity_id: str,
    entities: dict[str, Any],
    partial: bool = False,
    paths: list[dict[str, Any]] | None = None,
) -> None:
    cache.put(
        cache.key(
            httpx.Request(
                "GET",
                settings.sayari_api_base + f"/v1/supply_chain/upstream/{entity_id}",
                params={
                    "max_depth": settings.max_upstream_depth,
                    "limit": settings.upstream_limit,
                },
            ),
        ),
        json.dumps(
            {
                "filters": {},
                "explored_count": len(entities),
                "data": {"entities": entities, "paths": paths if paths is not None else []},
                "partial_results": partial,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    )


def _node(identifier: str, countries: int, factors: list[str]) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": "company",
        "label": f"节点 {identifier}",
        "translated_label": f"Translated {identifier}",
        "countries": [f"C{i}" for i in range(countries)],
        "risk_factors": factors,
    }


@pytest.fixture
def forbid_external_stages(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    send = Mock(side_effect=AssertionError("No external stage request"))
    monkeypatch.setattr(AuditedTransport, "_send", send)
    yield
    send.assert_not_called()


@pytest.fixture
def graph(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, ResponseCache]:
    configured, cache = _prepare(
        settings,
        tmp_path,
        {
            "list_3": [(f"Input {i}", f"s{i}") for i in range(1, 7)]
            + [("Duplicate input", "s1"), ("Unresolved input", None)],
            "another_portfolio": [(f"Additional {i}", f"t{i}") for i in range(1, 3)],
        },
    )
    monkeypatch.setattr(pipeline, "load_ontology", synthetic_ontology)
    nodes = {
        "n-top": _node("n-top", 1, ["sanctioned_synthetic", "ilab_forced_labor", "mystery"]),
        "n-boundary": _node("n-boundary", 2, ["military_synthetic", "sanctioned_synthetic"]),
        "h-broad": _node("h-broad", 3, ["ilab_forced_labor"]),
        "h-severe": _node("h-severe", 4, ["military_synthetic"]),
        "broad": _node("broad", 1, ["ilab_forced_labor"]),
        "solo": _node("solo", 1, ["military_synthetic"]),
    }
    for i in range(1, 7):
        keys = ["n-top"]
        if i <= 2:
            keys += ["n-boundary", "h-broad", "h-severe", "broad"]
        if i == 1:
            keys += ["solo"]
        _upstream(cache, configured, f"s{i}", {key: nodes[key] for key in keys}, partial=i == 2)
    # A shared ID in another portfolio cannot inflate headline membership.
    for i in range(1, 3):
        _upstream(cache, configured, f"t{i}", {"n-top": nodes["n-top"]})
    return configured, cache


@pytest.fixture
def findings(
    graph: tuple[Settings, ResponseCache], tmp_path: Path, forbid_external_stages: None
) -> Findings:
    return pipeline.run_pipeline(
        graph[0], all_sheets=True, offline=True, output_dir=tmp_path / "out"
    )

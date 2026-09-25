"""Exercise conformance diagnostics and optionally validate the full S0 corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.sdk_conformance import (
    classify,
    inspect_cache,
    inspect_response,
    inspect_tree,
    inventory_caches,
    main,
)


def upstream_payload() -> dict[str, Any]:
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


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"fields": {}, "data": []}, "resolution"),
        ({"id": "synthetic", "degree": 0}, "entity"),
        ({"partial_results": False, "explored_count": 0}, "upstream"),
        ({"data": []}, "unknown"),
        ([], "unknown"),
        ({"fields": {}, "data": [], "id": "ambiguous", "degree": 0}, "unknown"),
    ],
)
def test_shape_classification(payload: object, expected: str) -> None:
    # Only an unambiguous, supported response shape is matched to a conformance model.
    assert classify(payload) == expected


def test_path_counts_missing_ids_and_model_extra() -> None:
    # The structure counts include hops to missing entities and extra fields the model kept.
    payload = upstream_payload()
    # Duplicate records must count separately; nothing is deduplicated as a graph.
    payload["data"]["paths"] *= 2
    result = inspect_response(json.dumps(payload).encode(), "synthetic.json")
    assert result["pass"] is True
    assert result["upstream"] == {
        "paths": 2,
        "hops": 2,
        "components": 2,
        "entities": 1,
        "missing_hops": [
            {"path_index": 0, "hop_index": 0, "entity_id": "<redacted>"},
            {"path_index": 1, "hop_index": 0, "entity_id": "<redacted>"},
        ],
        "translated_label_model_extra": 1,
        "translated_label_nonnull": 1,
    }


def test_validation_failure_preserves_positions_without_disclosing_value() -> None:
    # Validation errors keep numeric list positions but redact everything taken from the source.
    payload = upstream_payload()
    payload["data"]["paths"][0]["path"][0]["components"][0]["hs_code"] = 1234
    result = inspect_response(json.dumps(payload).encode(), "synthetic.json")
    assert result["pass"] is False
    assert result["errors"] == [
        {
            "path": ["<redacted>", "<redacted>", 0, "<redacted>", 0, "<redacted>", 0, "<redacted>"],
            "type": "string_type",
            "value": "<redacted>",
        }
    ]
    assert "1234" not in json.dumps(result)


def test_missing_field_does_not_disclose_parent_payload() -> None:
    # A missing-field error never discloses the rest of the response.
    payload = upstream_payload()
    del payload["filters"]
    result = inspect_response(json.dumps(payload).encode(), "synthetic.json")
    assert result["pass"] is False
    assert result["errors"] == [{"path": ["<redacted>"], "type": "missing", "value": "<missing>"}]
    assert "Synthetic entity" not in json.dumps(result)


def test_invalid_json_is_not_disclosed() -> None:
    # Malformed JSON gives a fixed error message that contains none of the input.
    result = inspect_response(b'{"private": "DO NOT PRINT"', "synthetic.json")
    assert result["pass"] is False
    assert result["errors"] == [{"path": [], "type": "invalid_json"}]
    assert "DO NOT PRINT" not in json.dumps(result)


def test_cache_is_sorted_and_unchanged(tmp_path: Path) -> None:
    # Inspection lists results in a fixed order and never changes the cached bytes.
    raw = b'{"fields": {}, "data": []}'
    for key in ("f" * 64, "0" * 64):
        (tmp_path / f"{key}.json").write_bytes(raw)
    first = inspect_cache(tmp_path)
    assert first == inspect_cache(tmp_path)
    assert [row["file"] for row in first["results"]] == [f"{'0' * 64}.json", f"{'f' * 64}.json"]
    assert all(path.read_bytes() == raw for path in tmp_path.glob("*.json"))
    assert first["classes"] == {"resolution": {"files": 2, "passed": 2, "failed": 0}}


def test_empty_corpus_fails(tmp_path: Path) -> None:
    # An empty response corpus is an error, never a conformance pass.
    with pytest.raises(ValueError, match="No cache JSON files found"):
        inspect_cache(tmp_path)


def test_cli_reports_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # One invalid response fails the gate, but the later files are still checked.
    (tmp_path / f"{'0' * 64}.json").write_text('{"unclassified": true}')
    (tmp_path / f"{'f' * 64}.json").write_text('{"fields": {}, "data": []}')
    monkeypatch.setattr("sys.argv", ["sdk_conformance.py", str(tmp_path)])
    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["files"] == 2
    assert report["classes"]["resolution"]["passed"] == 1
    assert report["classes"]["unknown"]["failed"] == 1


def test_cached_responses(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    # Every response in the selected corpus validates against its SDK model.
    cache_arg = pytestconfig.getoption("sdk_cache_dir")
    if cache_arg is None:
        (tmp_path / f"{'0' * 64}.json").write_text('{"fields": {}, "data": []}')
        cache_dir = tmp_path
    else:
        cache_dir = Path(cache_arg)
    report = inspect_cache(cache_dir)
    failed = [row for row in report["results"] if not row["pass"]]
    assert failed == [], json.dumps(failed, sort_keys=True)
    summary = {key: value for key, value in report.items() if key != "results"}
    print(
        "\nCache conformance (explicit corpus):"
        if cache_arg
        else "\nCache conformance (synthetic):"
    )
    print(json.dumps(summary, sort_keys=True, indent=2))


def test_recursive_inventory_before_validation(tmp_path: Path) -> None:
    # A recursive scan counts every cache file first, before it parses any of them.
    for directory in ("phase-a/cache", "phase-b/seed", ".pytest_tmp/case/cache"):
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        (folder / f"{'a' * 64}.json").write_bytes(b"invalid json")
    excluded = tmp_path / ".venv" / "cache"
    excluded.mkdir(parents=True)
    (excluded / f"{'b' * 64}.json").write_bytes(b"excluded dependency")
    inventory = inventory_caches(tmp_path)
    assert inventory["total_directories"] == 3
    assert inventory["total_response_files"] == 3
    assert inventory["total_response_bytes"] == 36
    assert [row["directory"] for row in inventory["directories"]] == [
        ".pytest_tmp/case/cache",
        "phase-a/cache",
        "phase-b/seed",
    ]
    report = inspect_tree(tmp_path)
    assert report["inventory"] == inventory
    assert len(report["caches"]) == 3
    assert all(cache["results"][0]["pass"] is False for cache in report["caches"])
    # The same key in three cache generations is still three files, each validated on its own.
    assert sum(cache["files"] for cache in report["caches"]) == 3


def test_inventory_only_does_not_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Inventory-only mode counts the files and does not check them against the models.
    (tmp_path / f"{'a' * 64}.json").write_bytes(b"malformed")
    monkeypatch.setattr("sys.argv", ["sdk_conformance.py", str(tmp_path), "--inventory-only"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["total_response_files"] == 1


def test_all_cache_generations(pytestconfig: pytest.Config, tmp_path: Path) -> None:
    # Recursive inspection covers every cache generation it finds and changes none of them.
    root_arg = pytestconfig.getoption("sdk_cache_root")
    if root_arg is None:
        (tmp_path / f"{'a' * 64}.json").write_text('{"fields": {}, "data": []}')
        root = tmp_path
    else:
        root = Path(root_arg)
    before = inventory_caches(root)
    report = inspect_tree(root)
    assert report["inventory"] == before
    assert inventory_caches(root) == before
    assert sum(cache["files"] for cache in report["caches"]) == before["total_response_files"]
    if root_arg is not None:
        output = Path(pytestconfig.rootpath) / "data/processed/sdk-conformance-all.json"
        output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("\nExpanded cache validation:")
    print(
        json.dumps(
            {
                "directories": before["total_directories"],
                "files": before["total_response_files"],
                "bytes": before["total_response_bytes"],
                "passed": sum(
                    row["pass"] for cache in report["caches"] for row in cache["results"]
                ),
                "failed": sum(
                    not row["pass"] for cache in report["caches"] for row in cache["results"]
                ),
            },
            sort_keys=True,
            indent=2,
        )
    )
    # This is a measurement gate, not an assumption that historical synthetic fixtures satisfy the
    # SDK. Every outcome must be recorded, including failures.
    assert all("pass" in row for cache in report["caches"] for row in cache["results"])

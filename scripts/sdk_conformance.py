"""Measure SDK model conformance using read-only, sha256-keyed cache files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sayari.entity import GetEntityResponse
from sayari.resolution import ResolutionResponse
from sayari.supply_chain import (
    TradeTraversalComponent,
    TradeTraversalEntity,
    TradeTraversalPath,
    TradeTraversalPathSegment,
    UpstreamTradeTraversalResponse,
)

CACHE_FILENAME = re.compile(r"[0-9a-f]{64}\.json")

MODELS = {
    "resolution": ResolutionResponse,
    "entity": GetEntityResponse,
    "upstream": UpstreamTradeTraversalResponse,
}


def classify(payload: object) -> str:
    """Select only an unambiguous supported response model."""
    if not isinstance(payload, dict):
        return "unknown"
    matches = []
    if "fields" in payload and isinstance(payload.get("data"), list):
        matches.append("resolution")
    if "id" in payload and "degree" in payload:
        matches.append("entity")
    if "partial_results" in payload and "explored_count" in payload:
        matches.append("upstream")
    # Choose a model only when exactly one supported shape matches.
    return matches[0] if len(matches) == 1 else "unknown"


def upstream_counts(model: UpstreamTradeTraversalResponse) -> dict[str, Any]:
    """Count parsed structure and locate dangling hops by position."""
    missing = []
    hops = 0
    components = 0
    for path_index, path in enumerate(model.data.paths):
        assert isinstance(path, TradeTraversalPath)
        for hop_index, hop in enumerate(path.path):
            assert isinstance(hop, TradeTraversalPathSegment)
            hops += 1
            if hop.entity_id not in model.data.entities:
                # Keep just path and hop positions, plus a fixed placeholder for the missing ID.
                missing.append(
                    {
                        "path_index": path_index,
                        "hop_index": hop_index,
                        "entity_id": "<redacted>",
                    }
                )
            for component in hop.components:
                assert isinstance(component, TradeTraversalComponent)
                components += 1
    extra_labels = 0
    nonnull_labels = 0
    for entity_id in sorted(model.data.entities):
        entity = model.data.entities[entity_id]
        assert isinstance(entity, TradeTraversalEntity)
        # TradeTraversalEntity permits extra fields; inspect them without copying their values into
        # the report. Verified in sayari/supply_chain/types/trade_traversal_entity.py.
        extra = entity.model_extra or {}
        if "translated_label" in extra:
            extra_labels += 1
            nonnull_labels += extra["translated_label"] is not None
    return {
        "paths": len(model.data.paths),
        "hops": hops,
        "components": components,
        "entities": len(model.data.entities),
        "missing_hops": missing,
        "translated_label_model_extra": extra_labels,
        "translated_label_nonnull": nonnull_labels,
    }


def inspect_response(raw: bytes, filename: str) -> dict[str, Any]:
    """Validate cached bytes and emit redacted structural results."""
    result: dict[str, Any] = {"file": filename, "classification": "unknown", "pass": False}
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError):
        result["errors"] = [{"path": [], "type": "invalid_json"}]
        return result
    kind = classify(payload)
    result["classification"] = kind
    if kind == "unknown":
        result["errors"] = [{"path": [], "type": "unclassified_or_ambiguous_shape"}]
        return result
    try:
        model = MODELS[kind].model_validate(payload)
    except ValidationError as exc:
        # String parts of an error location can be mapping keys taken from the data, so keep only
        # numeric positions, error types and fixed placeholder values.
        result["errors"] = [
            {
                "path": [part if isinstance(part, int) else "<redacted>" for part in error["loc"]],
                "type": error["type"],
                "value": "<missing>" if error["type"] == "missing" else "<redacted>",
            }
            for error in exc.errors(include_url=False, include_context=False, include_input=False)
        ]
        return result
    result["pass"] = True
    if kind == "upstream":
        result["upstream"] = upstream_counts(model)
    return result


def inspect_cache(cache_dir: Path) -> dict[str, Any]:
    """Inspect and fingerprint an ordered response inventory."""
    files = sorted(cache_dir.glob("*.json"), key=lambda path: path.name)
    # Fail on an empty cache rather than report a pass that checked nothing.
    if not files:
        raise ValueError("No cache JSON files found")
    results = []
    inventory = hashlib.sha256()
    for path in files:
        # Only include JSON files with opaque, digest-based names in the response inventory.
        if CACHE_FILENAME.fullmatch(path.name) is None:
            raise ValueError("Cache contains a non-sha256 JSON filename")
        raw = path.read_bytes()
        # Include the file name in the ordered inventory fingerprint, not just its contents.
        inventory.update(path.name.encode("ascii"))
        inventory.update(hashlib.sha256(raw).digest())
        results.append(inspect_response(raw, path.name))
    classes: dict[str, dict[str, int]] = {}
    counts: Counter[str] = Counter()
    for result in results:
        kind = result["classification"]
        row = classes.setdefault(kind, {"files": 0, "passed": 0, "failed": 0})
        row["files"] += 1
        row["passed" if result["pass"] else "failed"] += 1
        for key, value in result.get("upstream", {}).items():
            counts[key] += len(value) if key == "missing_hops" else value
    return {
        "inventory_sha256": inventory.hexdigest(),
        "files": len(files),
        "classes": dict(sorted(classes.items())),
        "upstream_totals": dict(sorted(counts.items())),
        "results": results,
    }


def inventory_caches(root: Path) -> dict[str, Any]:
    """Find opaque response inventories without reading payloads."""
    excluded = {".git", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    directories: list[dict[str, Any]] = []
    for directory, children, filenames in os.walk(root):
        children[:] = sorted(
            name
            for name in children
            if name.lower() not in excluded and not name.lower().startswith(".venv")
        )
        if not any(CACHE_FILENAME.fullmatch(name) for name in filenames):
            continue
        current = Path(directory)
        files = sorted(current.iterdir(), key=lambda path: path.name)
        files = [path for path in files if path.is_file()]
        responses = [path for path in files if CACHE_FILENAME.fullmatch(path.name)]
        directories.append(
            {
                "directory": current.relative_to(root).as_posix(),
                "files": len(files),
                "json_files": sum(path.suffix.lower() == ".json" for path in files),
                "response_files": len(responses),
                "response_bytes": sum(path.stat().st_size for path in responses),
                "bytes": sum(path.stat().st_size for path in files),
            }
        )
    directories.sort(key=lambda row: row["directory"])
    if not directories:
        raise ValueError("No response cache directories found")
    return {
        "directories": directories,
        "total_directories": len(directories),
        "total_response_files": sum(row["response_files"] for row in directories),
        "total_response_bytes": sum(row["response_bytes"] for row in directories),
    }


def inspect_tree(root: Path) -> dict[str, Any]:
    """Inspect discovered caches without changing evidence."""
    inventory = inventory_caches(root)
    reports = []
    for row in inventory["directories"]:
        directory = row["directory"]
        report = inspect_cache(root / directory)
        reports.append({"directory": directory, **report})
    return {"inventory": inventory, "caches": reports}


def main() -> int:
    """Report local conformance or command failure safely."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Discover all response caches.")
    parser.add_argument(
        "--inventory-only", action="store_true", help="Report sizes before parsing."
    )
    args = parser.parse_args()
    try:
        if args.inventory_only:
            report = inventory_caches(args.cache_dir)
            # A successful inventory counts as success; it says nothing about model conformance.
            failed = False
        elif args.recursive:
            report = inspect_tree(args.cache_dir)
            failed = any(not row["pass"] for cache in report["caches"] for row in cache["results"])
        else:
            report = inspect_cache(args.cache_dir)
            failed = any(not row["pass"] for row in report["results"])
    except (OSError, ValueError) as exc:
        # Exception messages may contain private absolute paths.
        print(json.dumps({"error": type(exc).__name__}))
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

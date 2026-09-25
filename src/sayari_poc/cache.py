"""Deterministic raw-response storage; authentication never enters this cache."""

import hashlib
import json
import re
from pathlib import Path

import httpx


class ResponseCache:
    """Raw successful responses keyed by SDK-built request identity."""

    def __init__(self, cache_dir: Path) -> None:
        """Use root for raw response storage without creating it yet."""
        self.cache_dir = cache_dir

    def key(self, request: httpx.Request) -> str:
        """Hash request identity without losing repeated query values."""
        # Keep repeated query values; a dict would collapse them and let two requests share one key.
        canonical = json.dumps(
            [
                str(request.url.copy_with(path="/", query=None, fragment=None)).rstrip("/"),
                request.method,
                request.url.raw_path.split(b"?", 1)[0].decode("ascii"),
                sorted(request.url.params.multi_items()),
            ],
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        """Resolve only valid digest keys inside the cache directory."""
        # Only a lowercase SHA-256 hex digest may become a file name.
        if re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Invalid cache key: expected a SHA-256 hex digest")
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _validate(content: bytes) -> None:
        """Reject corrupt envelopes without exposing response contents."""
        try:
            value = json.loads(content)
        except ValueError:
            # Say the JSON is corrupt, but keep the response and parser details out of the error.
            raise ValueError("Malformed JSON in response cache") from None
        if not isinstance(value, dict):
            raise ValueError("Response cache must contain a JSON object")

    def get(self, key: str) -> bytes | None:
        """Return original response bytes, or None for a missing entry.

        Corrupt evidence raises ValueError rather than becoming a cache miss.
        """
        # Only a missing file counts as a cache miss.
        try:
            content = self._path(key).read_bytes()
        except FileNotFoundError:
            return None
        # A corrupt entry is an error; don't quietly fall through to a network request.
        self._validate(content)
        return content

    def put(self, key: str, content: bytes) -> None:
        """Publish validated raw bytes atomically.

        A failed write preserves the previous entry and removes the staging file.
        """
        path = self._path(key)
        self._validate(content)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and swap it in, so an interrupted refresh leaves the old response.
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

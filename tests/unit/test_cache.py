"""Built-request identity and byte-preserving, atomic cache storage."""

from pathlib import Path

import httpx
import pytest

from sayari_poc.cache import ResponseCache


def test_request_key_is_canonical_and_preserves_scope(tmp_path: Path) -> None:
    # Equivalent query encodings share one key; a different method, host, path or query does not.
    cache = ResponseCache(tmp_path)
    first = httpx.Request(
        "GET", "https://example.invalid/resource", params=[("z", "2"), ("a", "1")]
    )
    reordered = httpx.Request("GET", "https://example.invalid/resource?a=1&z=2")
    assert cache.key(first) == cache.key(reordered)
    for request in (
        httpx.Request("POST", first.url),
        httpx.Request("GET", "https://other.invalid/resource?a=1&z=2"),
        httpx.Request("GET", "https://example.invalid/other?a=1&z=2"),
        httpx.Request("GET", "https://example.invalid/resource?a=1&z=3"),
        httpx.Request("GET", "https://example.invalid/resource?a=1&a=1&z=2"),
    ):
        assert cache.key(request) != cache.key(first)
    assert cache.key(httpx.Request("GET", "https://example.invalid/?name=%C3%A9")) == cache.key(
        httpx.Request("GET", "https://example.invalid/", params={"name": "é"})
    )


def test_raw_bytes_round_trip_and_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cache stores exact bytes, and a failed write leaves the old entry in place.
    cache = ResponseCache(tmp_path / "cache")
    key = cache.key(httpx.Request("GET", "https://example.invalid/resource"))
    content = '{\n  "label": "测试", "unused": [3,2,1]\n}\n'.encode()
    assert cache.get(key) is None
    cache.put(key, content)
    assert ResponseCache(cache.cache_dir).get(key) == content

    def interrupted(self: Path, target: Path) -> Path:
        raise OSError("simulated interruption")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError):
        cache.put(key, b"{}")
    assert cache.get(key) == content
    assert not list(cache.cache_dir.glob("*.tmp"))


@pytest.mark.parametrize("key", ["../escape", "a" * 63, "G" * 64, "/absolute"])
def test_cache_keys_cannot_escape_directory(tmp_path: Path, key: str) -> None:
    # A malformed digest key is rejected before it can reach a path outside the cache.
    cache = ResponseCache(tmp_path)
    with pytest.raises(ValueError, match="key"):
        cache.put(key, b"{}")
    with pytest.raises(ValueError, match="key"):
        cache.get(key)


@pytest.mark.parametrize("content", [b"{broken", b"[]", b"null", b"42", b"\xff"])
def test_invalid_cache_write_is_rejected_without_files(tmp_path: Path, content: bytes) -> None:
    # An invalid response body is rejected and leaves no cache file behind.
    cache = ResponseCache(tmp_path)
    with pytest.raises(ValueError):
        cache.put("a" * 64, content)
    assert not list(tmp_path.iterdir())

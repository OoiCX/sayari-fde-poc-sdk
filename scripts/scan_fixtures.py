"""Scan local evidence without printing matched text or credential values."""

import argparse
import base64
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

from sayari_poc.config import load_settings

# Words that need a per-file review when found. A hit on its own doesn't prove a secret leaked.
MARKERS = (
    "bearer",
    "authorization",
    "access_token",
    "refresh_token",
    "client_secret",
    "client_id",
    "eyj",
    "token_type",
    "expires_in",
    "grant_type",
)


def scan_content(content: bytes, secrets: list[str], root: Path) -> list[str]:
    """Detect finding categories without returning matched contents."""
    lowered = content.lower()
    hits = [marker for marker in MARKERS if marker.encode() in lowered]
    for secret in secrets:
        if not secret:
            continue
        # Look for the configured secret as raw text and in its JSON, HTML, URL and base64 forms.
        forms = (
            secret,
            json.dumps(secret)[1:-1],
            html.escape(secret),
            quote(secret, safe=""),
            base64.b64encode(secret.encode()).decode(),
        )
        if any(form.encode() in content for form in forms):
            # A real configured secret always fails the scan; no review can clear it.
            hits.append("configured_credential")
            break
    paths = ("C:\\Users\\", "/c/Users/", str(root.resolve()), root.resolve().as_posix())
    # Catch private absolute paths, as written or JSON-escaped, ignoring case.
    if any(
        form.lower().encode() in lowered
        for path in paths
        for form in (path, json.dumps(path)[1:-1])
    ):
        hits.append("private_absolute_path")
    return hits


def scan_file(path: Path, secrets: list[str], root: Path) -> list[str]:
    """Scan raw or decompressed evidence for safe finding categories."""
    try:
        content = path.read_bytes()
        if path.suffix == ".gz":
            content = gzip.decompress(content)
        if path.suffix == ".xlsx":
            with ZipFile(io.BytesIO(content)) as archive:
                # Scan every archive member together, in sorted order, so results are repeatable.
                content = b"\n".join(archive.read(name) for name in sorted(archive.namelist()))
    except (OSError, EOFError, ZlibError, BadZipFile):
        return ["unreadable_file"]
    return scan_content(content, secrets, root)


def apply_review(path: Path, hits: list[str], review: dict[str, Any]) -> list[str]:
    """Accept only exact content and category review matches."""
    # A review can clear marker words and private paths, but never a real configured secret.
    permitted = {*MARKERS, "private_absolute_path"}
    expected = review.get("categories", [])
    if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
        return ["invalid_review"]
    if not set(expected) <= permitted or "configured_credential" in hits:
        return hits or ["invalid_review"]
    # The review only holds while the file's raw bytes still match the approved SHA-256 digest.
    if review.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        return ["review_content_changed"]
    # The categories found must match the reviewed categories exactly.
    if set(hits) != set(expected):
        return hits or ["review_categories_changed"]
    return []


def main() -> int:
    """Fail scans with unreviewed findings or unreadable inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--reviewed-markers", type=Path)
    parser.add_argument(
        "--app-credentials",
        action="store_true",
        help="also check for the Sayari credentials the app would load, including from .env",
    )
    args = parser.parse_args()
    reviews: dict[str, Any] = {}
    if args.reviewed_markers is not None:
        try:
            document = json.loads(args.reviewed_markers.read_text(encoding="utf-8"))
            if (
                not isinstance(document, dict)
                or document.get("path_key") != "sha256-posix-relative"
            ):
                raise ValueError
            records = document.get("reviews")
            if not isinstance(records, dict) or not all(
                isinstance(key, str)
                and len(key) == 64
                and all(char in "0123456789abcdef" for char in key)
                for key in records
            ):
                raise ValueError
            reviews = records
        except (OSError, ValueError):
            # Report a missing or invalid review file without revealing paths or parser details.
            print("FAIL: marker review unavailable or invalid")
            return 1
    # Read the two configured credentials only so we can check whether they leaked.
    secrets = [os.environ.get(name, "") for name in ("SAYARI_CLIENT_ID", "SAYARI_CLIENT_SECRET")]
    if args.app_credentials:
        # Load them the way the app does, so credentials kept only in .env are checked too.
        try:
            settings = load_settings()
        except (OSError, ValueError):
            # Don't echo the settings error: it can quote configured values.
            print("FAIL: app settings could not be loaded")
            return 1
        secret = settings.sayari_client_secret
        secrets = [settings.sayari_client_id or "", secret.get_secret_value() if secret else ""]
    configured = sum(bool(value) for value in secrets)
    # Say whether the credential check ran, so an empty check can't pass unnoticed.
    print(
        f"Credential check: {configured} configured value(s)"
        if configured
        else "Credential check: skipped, no credentials configured"
    )
    paths = args.paths
    files: set[Path] = set()
    missing = False
    for path in paths:
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        elif path.is_file():
            files.add(path)
        else:
            print("FAIL (missing scan input; path withheld)")
            missing = True
    failed = missing
    reviewed_count = 0
    for path in sorted(files):
        hits = scan_file(path, secrets, Path.cwd())
        try:
            display = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            # Show only the file name when the file is outside the root.
            display = path.name
            # Files outside the root never inherit a review, even if their name and bytes match.
            review = None
        else:
            review = reviews.get(hashlib.sha256(display.encode("utf-8")).hexdigest())
        if review is not None:
            if not isinstance(review, dict):
                hits = ["invalid_review"]
            else:
                hits = apply_review(path, hits, review)
                reviewed_count += not hits
        failed |= bool(hits)
        # Print the safe display path and category names, never matched text or secret values.
        print(f"{display}: FAIL ({', '.join(hits)})" if hits else f"{display}: PASS")
    print(f"Scanned {len(files)} files; reviewed marker files={reviewed_count}; failed={failed}")
    return int(failed or not files)


if __name__ == "__main__":
    raise SystemExit(main())

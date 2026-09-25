"""Evidence scans detect encoded secrets without returning their contents."""

import base64
import gzip
import hashlib
import html
import json
import runpy
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

import pytest

from scripts import sdk_conformance
from tests.unit.test_sdk_conformance import upstream_payload

SCANNER = runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "scan_fixtures.py"))


@pytest.mark.parametrize("location", ["nested_value", "dynamic_key"])
def test_conformance_cli_redacts_input_values_and_dynamic_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    location: str,
) -> None:
    # Conformance diagnostics redact both response values and dynamic mapping keys.
    private = "synthetic-private-value"
    private_path = "C:\\Users\\synthetic-private\\evidence.json"
    payload = upstream_payload()
    if location == "nested_value":
        payload["data"]["paths"][0]["path"][0]["components"][0]["hs_code"] = {
            "authorization": {"access_token": private, "path": private_path}
        }
    else:
        payload["data"]["entities"] = {private_path: {"id": private, "label": []}}
    (tmp_path / f"{'a' * 64}.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["sdk_conformance.py", str(tmp_path)])
    assert sdk_conformance.main() == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert private not in output
    assert private_path not in output and json.dumps(private_path)[1:-1] not in output
    assert SCANNER["scan_content"](output.encode(), [private], tmp_path) == []
    errors = json.loads(captured.out)["results"][0]["errors"]
    assert errors and all(error["type"] for error in errors)
    assert all(error["value"] in {"<missing>", "<redacted>"} for error in errors)
    if location == "nested_value":
        assert [part for part in errors[0]["path"] if isinstance(part, int)] == [0, 0, 0]


def test_out_of_root_same_name_cannot_inherit_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A file outside the repository with the same name does not inherit the repository's review.
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(root)
    approved = root / "approved.txt"
    approved.write_bytes(b"Authorization")
    external = tmp_path / approved.name
    external.write_bytes(approved.read_bytes())
    index = root / "review.json"
    index.write_text(
        json.dumps(
            {
                "path_key": "sha256-posix-relative",
                "reviews": {
                    hashlib.sha256(b"approved.txt").hexdigest(): {
                        "sha256": hashlib.sha256(approved.read_bytes()).hexdigest(),
                        "categories": ["authorization"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv", ["scan_fixtures.py", "--reviewed-markers", str(index), str(approved)]
    )
    assert SCANNER["main"]() == 0
    assert "reviewed marker files=1; failed=False" in capsys.readouterr().out
    monkeypatch.setattr(
        "sys.argv", ["scan_fixtures.py", "--reviewed-markers", str(index), str(external)]
    )
    assert SCANNER["main"]() == 1
    output = capsys.readouterr().out
    assert "approved.txt: FAIL (authorization)" in output
    assert "reviewed marker files=0; failed=True" in output
    assert str(tmp_path) not in output


@pytest.mark.parametrize(
    "encode",
    [
        str,
        lambda s: json.dumps(s)[1:-1],
        html.escape,
        lambda s: quote(s, safe=""),
        lambda s: base64.b64encode(s.encode()).decode(),
    ],
)
def test_encoded_secret_is_detected_without_disclosure(tmp_path: Path, encode: object) -> None:
    # An encoded secret is reported by category only, never by the matched value.
    secret = 'synthetic-<private>"/value'
    assert callable(encode)
    hits = SCANNER["scan_content"](encode(secret).encode(), [secret], tmp_path)
    assert "configured_credential" in hits
    assert secret not in str(hits)


@pytest.mark.parametrize("marker", ["Authorization", "Bearer", "access_token", "token_type"])
def test_oauth_markers_are_case_insensitive(tmp_path: Path, marker: str) -> None:
    # OAuth markers are detected whatever their letter case.
    assert marker.lower() in SCANNER["scan_content"](marker.encode(), [], tmp_path)


def test_decompressed_snapshot_and_escaped_paths(tmp_path: Path) -> None:
    # Compressed evidence is scanned after decompression, and escaped paths are still caught.
    path = tmp_path / "snapshot.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"path": "C:\\Users\\private"}).encode()))
    assert SCANNER["scan_file"](path, [], tmp_path) == ["private_absolute_path"]


def test_clean_payload_and_corrupt_snapshot(tmp_path: Path) -> None:
    # Clean evidence passes, and a corrupt compressed file is reported as a failure.
    assert SCANNER["scan_content"](b'{"data": [], "partial_results": true}', [], tmp_path) == []
    path = tmp_path / "broken.gz"
    path.write_bytes(b"invalid")
    assert SCANNER["scan_file"](path, [], tmp_path) == ["unreadable_file"]


def test_review_is_bound_to_content_and_categories(tmp_path: Path) -> None:
    # A review approves only the exact bytes and the exact set of categories it recorded.
    path = tmp_path / "audit.md"
    content = b"Documentation mentions Authorization"
    path.write_bytes(content)
    review = {"sha256": hashlib.sha256(content).hexdigest(), "categories": ["authorization"]}
    assert SCANNER["apply_review"](path, ["authorization"], review) == []
    assert SCANNER["apply_review"](path, ["authorization", "access_token"], review)
    path.write_bytes(content + b" modified")
    assert SCANNER["apply_review"](path, ["authorization"], review) == ["review_content_changed"]


def test_review_cannot_exempt_configured_credentials(tmp_path: Path) -> None:
    # No review can waive the disclosure of a configured secret.
    path = tmp_path / "audit.md"
    content = b"synthetic secret"
    path.write_bytes(content)
    review = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "categories": ["configured_credential"],
    }
    assert SCANNER["apply_review"](path, ["configured_credential"], review) == [
        "configured_credential"
    ]


def test_scanner_inspects_workbook_contents(tmp_path: Path) -> None:
    # The scanner looks for credentials inside the workbook's archive contents.
    path = tmp_path / "fixture.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", "synthetic-secret")
    assert SCANNER["scan_file"](path, ["synthetic-secret"], tmp_path) == ["configured_credential"]


def test_scanner_requires_explicit_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # The scanner refuses to run unless it is given explicit paths to scan.
    monkeypatch.setattr("sys.argv", ["scan_fixtures.py"])
    with pytest.raises(SystemExit) as error:
        SCANNER["main"]()
    assert error.value.code == 2


def test_recursive_scan_ignores_bytecode_but_still_checks_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bytecode caches are skipped during a recursive scan, but ordinary evidence files never are.
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "clean.json").write_bytes(b"{}")
    cache = fixtures / "package" / "__pycache__"
    cache.mkdir(parents=True)
    bytecode = cache / "generated.pyc"
    bytecode.write_bytes(b"Authorization")
    monkeypatch.setattr("sys.argv", ["scan_fixtures.py", str(fixtures)])

    assert SCANNER["main"]() == 0
    output = capsys.readouterr().out
    assert "Scanned 1 files; reviewed marker files=0; failed=False" in output
    assert "generated.pyc" not in output
    assert bytecode.exists()

    evidence = fixtures / "__pycache__-evidence"
    evidence.mkdir()
    (evidence / "retained.json").write_bytes(b"Authorization")
    assert SCANNER["main"]() == 1
    output = capsys.readouterr().out
    assert "retained.json: FAIL (authorization)" in output
    assert "Scanned 2 files; reviewed marker files=0; failed=True" in output

    monkeypatch.setattr("sys.argv", ["scan_fixtures.py", str(bytecode)])
    assert SCANNER["main"]() == 1
    assert "generated.pyc: FAIL (authorization)" in capsys.readouterr().out


def test_digest_review_is_bound_to_the_exact_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An identical file at a different relative path needs its own review.
    monkeypatch.chdir(tmp_path)
    approved = tmp_path / "approved.txt"
    approved.write_bytes(b"Authorization")
    index = tmp_path / "review.json"
    index.write_text(
        json.dumps(
            {
                "path_key": "sha256-posix-relative",
                "reviews": {
                    hashlib.sha256(b"approved.txt").hexdigest(): {
                        "sha256": hashlib.sha256(approved.read_bytes()).hexdigest(),
                        "categories": ["authorization"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv", ["scan_fixtures.py", "--reviewed-markers", str(index), str(approved)]
    )
    assert SCANNER["main"]() == 0
    assert "reviewed marker files=1; failed=False" in capsys.readouterr().out
    relocated = tmp_path / "relocated.txt"
    relocated.write_bytes(approved.read_bytes())
    monkeypatch.setattr(
        "sys.argv", ["scan_fixtures.py", "--reviewed-markers", str(index), str(relocated)]
    )
    assert SCANNER["main"]() == 1
    assert "relocated.txt: FAIL (authorization)" in capsys.readouterr().out
    approved.write_bytes(b"Authorization changed")
    monkeypatch.setattr(
        "sys.argv", ["scan_fixtures.py", "--reviewed-markers", str(index), str(approved)]
    )
    assert SCANNER["main"]() == 1
    assert "review_content_changed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"path_key": "unknown", "reviews": {}},
        {"path_key": "sha256-posix-relative", "reviews": {"not-a-digest": {}}},
    ],
)
def test_invalid_review_index_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: object
) -> None:
    # A malformed review index approves nothing, so the scan fails.
    evidence = tmp_path / "clean.txt"
    evidence.write_bytes(b"clean")
    index = tmp_path / "review.json"
    index.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["scan_fixtures.py", "--reviewed-markers", str(index), str(evidence)]
    )
    assert SCANNER["main"]() == 1

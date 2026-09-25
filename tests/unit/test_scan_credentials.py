"""The fixture scanner's credential check, including credentials kept only in .env."""

import runpy
from pathlib import Path

import pytest

from sayari_poc.config import Settings

SCANNER = runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "scan_fixtures.py"))
SYNTHETIC_ID = "synthetic-dotenv-identifier"
SYNTHETIC_SECRET = "synthetic-dotenv-only-value"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Work in an empty folder with no Sayari settings in the environment, so only the .env each
    # test writes can supply credentials, and the real project .env is never read.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_dotenv(folder: Path, text: str) -> None:
    (folder / ".env").write_text(text, encoding="utf-8")


def _scan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    monkeypatch.setattr("sys.argv", ["scan_fixtures.py", *args])
    code: int = SCANNER["main"]()
    return code, capsys.readouterr().out


def test_app_credentials_catches_a_leak_of_a_dotenv_only_secret(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With --app-credentials, a credential kept only in .env is still caught when it leaks.
    _write_dotenv(
        workspace, f"SAYARI_CLIENT_ID={SYNTHETIC_ID}\nSAYARI_CLIENT_SECRET={SYNTHETIC_SECRET}\n"
    )
    (workspace / "leak.txt").write_text(f"payload {SYNTHETIC_SECRET}", encoding="utf-8")
    code, output = _scan(monkeypatch, capsys, "--app-credentials", "leak.txt")
    assert code == 1
    assert "Credential check: 2 configured value(s)" in output
    assert "leak.txt: FAIL (configured_credential)" in output
    assert SYNTHETIC_SECRET not in output and SYNTHETIC_ID not in output


def test_without_the_flag_dotenv_is_not_read_and_the_skip_is_reported(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without --app-credentials the scanner never reads .env, and says the check was skipped
    # rather than passing silently.
    _write_dotenv(workspace, f"SAYARI_CLIENT_SECRET={SYNTHETIC_SECRET}\n")
    (workspace / "leak.txt").write_text(f"payload {SYNTHETIC_SECRET}", encoding="utf-8")
    code, output = _scan(monkeypatch, capsys, "leak.txt")
    assert code == 0
    assert "Credential check: skipped, no credentials configured" in output
    assert "leak.txt: PASS" in output


def test_environment_credentials_are_checked_without_the_flag(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Credentials exported as environment variables are checked with or without the flag.
    monkeypatch.setenv("SAYARI_CLIENT_SECRET", SYNTHETIC_SECRET)
    (workspace / "leak.txt").write_text(f"payload {SYNTHETIC_SECRET}", encoding="utf-8")
    code, output = _scan(monkeypatch, capsys, "leak.txt")
    assert code == 1
    assert "Credential check: 1 configured value(s)" in output
    assert "leak.txt: FAIL (configured_credential)" in output


def test_unloadable_settings_fail_without_echoing_their_values(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An invalid .env stops the scan, and the error never quotes what the file contains.
    _write_dotenv(workspace, "SDK_MAX_RETRIES=synthetic-not-a-number\n")
    (workspace / "clean.txt").write_text("nothing sensitive", encoding="utf-8")
    code, output = _scan(monkeypatch, capsys, "--app-credentials", "clean.txt")
    assert code == 1
    assert "FAIL: app settings could not be loaded" in output
    assert "synthetic-not-a-number" not in output

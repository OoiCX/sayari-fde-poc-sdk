"""Environment settings and the committed offline defaults."""

import re
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from sayari_poc.config import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_offline_defaults_and_env_example_cover_every_setting() -> None:
    # Offline replay runs without credentials by default, and .env.example documents every setting.
    settings = load_settings()
    assert settings.sayari_client_id is None and settings.sayari_client_secret is None
    assert settings.cache_dir == Path("data/cache")
    assert settings.entity_file_path == Path("data/input/Sayari_Interview_Exercise_List.xlsx")
    assert settings.sdk_max_retries == 5 and settings.sdk_timeout_seconds == 60
    assert settings.declared_generated_at is None
    keys = set(re.findall(r"^([A-Z_0-9]+)=", (ROOT / ".env.example").read_text(), flags=re.M))
    assert keys == {name.upper() for name in Settings.model_fields}
    assert not {"max_pages", "enable_narration", "anthropic_api_key"} & Settings.model_fields.keys()


def test_dotenv_unicode_and_environment_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    # Environment variables win over .env, and a Unicode path in .env loads intact.
    Path(".env").write_text("ENTITY_FILE_PATH=测试.xlsx\nCALL_BUDGET=12\n", encoding="utf-8")
    monkeypatch.setenv("CALL_BUDGET", "25")
    settings = load_settings()
    assert settings.entity_file_path == Path("测试.xlsx") and settings.call_budget == 25


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CALL_BUDGET", "0"),
        ("MAX_UPSTREAM_DEPTH", "5"),
        ("UPSTREAM_LIMIT", "-1"),
        ("HUB_MAX_COUNTRIES", "invalid"),
        ("SDK_MAX_RETRIES", "-1"),
        ("SDK_TIMEOUT_SECONDS", "0"),
        ("SDK_TIMEOUT_SECONDS", "nan"),
    ],
)
def test_invalid_bounds_fail_safely(key: str, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # An out-of-range setting fails validation without leaking the client secret into the error.
    monkeypatch.setenv(key, value)
    monkeypatch.setenv("SAYARI_CLIENT_SECRET", "synthetic-hidden-value")
    with pytest.raises(ValidationError) as caught:
        load_settings()
    assert "synthetic-hidden-value" not in str(caught.value)


def test_empty_optional_values_and_explicit_zero_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty optional setting counts as unset, and zero retries is still a valid choice.
    monkeypatch.setenv("SAYARI_CA_BUNDLE", "")
    monkeypatch.setenv("DECLARED_GENERATED_AT", "")
    monkeypatch.setenv("SDK_MAX_RETRIES", "0")
    settings = load_settings()
    assert settings.sayari_ca_bundle is None and settings.sdk_max_retries == 0
    assert settings.declared_generated_at is None


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-19T12:38:32.773832+00:00",
        "2026-09-19T20:38:32.773832+08:00",
        "2026-09-19T12:38:32.773832Z",
    ],
)
def test_declared_generated_at_loads_verbatim(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # A declared output timestamp is kept exactly as written, offset spelling included.
    monkeypatch.setenv("DECLARED_GENERATED_AT", value)
    assert load_settings().declared_generated_at == value


@pytest.mark.parametrize("value", ["malformed-declared-time", "2026-09-19T12:38:32.773832"])
def test_invalid_declared_generated_at_rejected_at_startup(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A malformed or offset-free declared time fails at startup, and the error does not echo it.
    monkeypatch.setenv("DECLARED_GENERATED_AT", value)
    with pytest.raises(ValidationError) as caught:
        load_settings()
    assert [error["loc"] for error in caught.value.errors()] == [("declared_generated_at",)]
    assert value not in str(caught.value)


def test_source_date_epoch_is_not_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    # SOURCE_DATE_EPOCH is ignored, so it cannot quietly set the output timestamp.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1790000000")
    assert load_settings().declared_generated_at is None


def test_typing_marker_is_packaged_and_sdk_is_pinned() -> None:
    # The package ships its py.typed marker and pins the approved SDK version.
    source = ROOT / "src/sayari_poc"
    assert (source / "py.typed").read_bytes() == b""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "py.typed" in project["tool"]["setuptools"]["package-data"]["sayari_poc"]
    assert "sayari==0.1.43" in project["project"]["dependencies"]

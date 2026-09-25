"""Typed configuration loaded from the environment and an optional local .env."""

from datetime import datetime
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with environment-over-dotenv precedence."""

    # Settings can hold credentials, so validation errors must never echo the input.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        extra="ignore",
    )

    # Optional because offline replay must work without credentials; kept out of repr.
    sayari_client_id: str | None = Field(default=None, min_length=1, repr=False)
    # SecretStr also masks credentials in settings representations.
    sayari_client_secret: SecretStr | None = Field(default=None, min_length=1)
    sayari_api_base: str = "https://api.sayari.com"
    entity_file_path: Path = Path("data/input/Sayari_Interview_Exercise_List.xlsx")
    call_budget: int = Field(default=400, gt=0)
    cache_dir: Path = Path("data/cache")
    duckdb_path: Path = Path("data/processed/analysis.duckdb")
    # A fixed timestamp to declare, so fresh runs produce byte-identical artifacts.
    declared_generated_at: str | None = None
    max_upstream_depth: int = Field(default=3, gt=0, le=4)
    upstream_limit: int = Field(default=500, gt=0)
    hub_max_countries: int = Field(default=10, gt=0)
    sayari_ca_bundle: str | None = None
    sdk_max_retries: int = Field(default=5, ge=0)
    sdk_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @field_validator("declared_generated_at")
    @classmethod
    def validate_declared_generated_at(cls, value: str | None) -> str | None:
        """Validate a declared instant without changing its spelling."""
        if value is not None:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                raise ValueError("Must be an ISO-8601 datetime with a timezone offset") from None
            # Require a timezone so the declared time is one unambiguous instant.
            if parsed.tzinfo is None:
                raise ValueError("Must be an ISO-8601 datetime with a timezone offset")
        # Return it as written: reformatting an equivalent instant would still change the bytes.
        return value


def load_settings() -> Settings:
    """Load and validate startup settings."""
    return Settings()

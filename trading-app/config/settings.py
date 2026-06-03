"""
Application settings, loaded from environment variables / .env file.

Usage
-----
    from config.settings import get_settings

    settings = get_settings()
    print(settings.paper_trading)
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Phase(str, Enum):
    PHASE_1 = "PHASE_1"
    PHASE_2 = "PHASE_2"


class Settings(BaseSettings):
    """All runtime configuration for the trading application.

    Values are read (in priority order) from:
    1. Environment variables
    2. A ``.env`` file in the project root
    3. The defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Trading Mode ──────────────────────────────────────────────────────────
    paper_trading: bool = True

    # ── Capital ───────────────────────────────────────────────────────────────
    initial_capital_gbp: float = 100.0

    # ── Kraken (primary UK-compliant crypto broker) ───────────────────────────
    kraken_api_key: str = ""
    kraken_private_key: str = ""

    # ── Coinbase Advanced Trade (backup crypto broker) ────────────────────────
    coinbase_api_key: str = ""
    coinbase_private_key: str = ""

    # ── Interactive Brokers ───────────────────────────────────────────────────
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1

    # ── Market Data ───────────────────────────────────────────────────────────
    polygon_api_key: str = ""
    news_api_key: str = ""

    # ── Notifications ─────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Database ──────────────────────────────────────────────────────────────
    db_password: str = "change_me_in_production"
    timescaledb_url: str = (
        "postgresql+asyncpg://trader:change_me_in_production@localhost:5432/trading"
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Risk Management ───────────────────────────────────────────────────────
    max_daily_loss_pct: float = 0.20
    max_drawdown_pct: float = 0.35
    max_position_size_pct: float = 0.25
    # Absolute minimum equity in GBP — below this, all trading is halted.
    ruin_floor_gbp: float = 20.0
    # Minimum acceptable reward-to-risk ratio for a trade entry.
    min_rr_ratio: float = 2.0
    # Maximum acceptable bid/ask spread in basis points before rejecting a trade.
    max_spread_bps: float = 50.0

    # ── Growth Phases ─────────────────────────────────────────────────────────
    phase1_target_gbp: float = 10_000.0
    phase2_target_gbp: float = 1_000_000.0

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @field_validator("max_daily_loss_pct", "max_drawdown_pct", "max_position_size_pct")
    @classmethod
    def _validate_pct(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Percentage value must be between 0 (exclusive) and 1 (inclusive), got {v}")
        return v

    @model_validator(mode="after")
    def _force_paper_trading_when_no_key(self) -> "Settings":
        """Ensure paper_trading is True when no broker API key is configured."""
        if not self.kraken_api_key.strip() and not self.coinbase_api_key.strip():
            object.__setattr__(self, "paper_trading", True)
        return self

    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def current_phase(self) -> Phase:
        """Return the current growth phase based on initial capital vs targets.

        Phase 1: equity is below the Phase-1 target.
        Phase 2: equity has reached or exceeded the Phase-1 target.
        """
        if self.initial_capital_gbp < self.phase1_target_gbp:
            return Phase.PHASE_1
        return Phase.PHASE_2

    @property
    def is_paper_trading(self) -> bool:
        """Convenience alias kept for readability in strategy code."""
        return self.paper_trading

    @property
    def sync_db_url(self) -> str:
        """Synchronous (psycopg2) version of the database URL, for Alembic."""
        return self.timescaledb_url.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton.

    The result is cached after the first call so that the .env file is
    parsed only once per process.  Call ``get_settings.cache_clear()`` in
    tests when you need a fresh instance.
    """
    return Settings()

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    fmp_api_key: str = os.getenv("FMP_API_KEY", "")
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openrouter_base: str = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")


@dataclass
class ModelConfig:
    var_max_lags: int = 5
    var_trend: str = "n"
    impulse_response_horizon: int = 12
    tick_test_window: int = 15
    quote_test_window: int = 15


@dataclass
class LiquidityConfig:
    redemption_notice_periods: list = None
    redemption_frequencies: list = None
    lockup_periods: list = None
    fund_age_threshold: int = 24
    aum_thresholds: list = None

    def __post_init__(self):
        if self.redemption_notice_periods is None:
            self.redemption_notice_periods = [0, 30, 60, 90]
        if self.redemption_frequencies is None:
            self.redemption_frequencies = ["daily", "weekly", "monthly", "quarterly"]
        if self.lockup_periods is None:
            self.lockup_periods = [0, 3, 6, 12]
        if self.aum_thresholds is None:
            self.aum_thresholds = [1e6, 1e7, 1e8, 1e9]


@dataclass
class StrategyConfig:
    alpha_weight: float = 0.25
    plausibility_weight: float = 0.15
    technical_profile_weight: float = 0.15
    operational_framework_weight: float = 0.10
    portfolio_risk_weight: float = 0.15
    market_crowding_weight: float = 0.10
    sector_capacity_weight: float = 0.10


@dataclass
class Settings:
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    cache_dir: str = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    results_dir: str = os.path.join(os.path.dirname(__file__), "..", "results")


settings = Settings()
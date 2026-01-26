"""Market risk domain - VaR, CVaR, and volatility analytics."""

from .router import router
from .service import (
    calculate_var,
    calculate_cvar,
    calculate_volatility_metrics,
    calculate_liquidity_metrics,
    generate_market_risk_report,
    Z_SCORES,
    CVAR_MULTIPLIERS,
)
from .schemas import (
    # Enums
    RiskRating,
    # Core metrics
    CryptoVolatilityMetrics,
    LiquidityMetrics,
    MarketRiskReport,
    # Request/Response
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    MarketIntelligenceResponse,
    VaRCalculationRequest,
    VaRCalculationResponse,
)

__all__ = [
    # Router
    "router",
    # Service functions
    "calculate_var",
    "calculate_cvar",
    "calculate_volatility_metrics",
    "calculate_liquidity_metrics",
    "generate_market_risk_report",
    # Constants
    "Z_SCORES",
    "CVAR_MULTIPLIERS",
    # Enums
    "RiskRating",
    # Core metrics
    "CryptoVolatilityMetrics",
    "LiquidityMetrics",
    "MarketRiskReport",
    # Request/Response
    "RiskAssessmentRequest",
    "RiskAssessmentResponse",
    "MarketIntelligenceResponse",
    "VaRCalculationRequest",
    "VaRCalculationResponse",
]

"""Pydantic schemas for market risk analytics - VaR, CVaR, volatility, liquidity."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class RiskRating(str, Enum):
    """Risk rating classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# =============================================================================
# Core Metrics Models
# =============================================================================


class CryptoVolatilityMetrics(BaseModel):
    """Volatility metrics for a cryptocurrency asset.

    Includes historical volatility, VaR, CVaR, and correlation metrics
    essential for portfolio risk management.
    """
    asset: str = Field(..., description="Asset identifier (e.g., BTC, ETH)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Volatility measures
    rolling_volatility_30d: float = Field(
        ..., ge=0, description="30-day rolling annualized volatility"
    )
    rolling_volatility_90d: float = Field(
        0.0, ge=0, description="90-day rolling annualized volatility"
    )

    # Value at Risk (parametric, assuming normal distribution)
    var_95: float = Field(..., description="95% 1-day Value at Risk (percentage)")
    var_99: float = Field(..., description="99% 1-day Value at Risk (percentage)")

    # Conditional VaR (Expected Shortfall)
    cvar_95: float = Field(..., description="95% Expected Shortfall (percentage)")
    cvar_99: float = Field(0.0, description="99% Expected Shortfall (percentage)")

    # Drawdown metrics
    max_drawdown: float = Field(..., ge=0, le=1, description="Maximum drawdown observed (0-1)")
    current_drawdown: float = Field(0.0, ge=0, le=1, description="Current drawdown from recent high")

    # Correlation to major assets
    correlation_btc: float = Field(0.0, ge=-1, le=1, description="Correlation to Bitcoin")
    correlation_eth: float = Field(0.0, ge=-1, le=1, description="Correlation to Ethereum")
    correlation_spy: float = Field(0.0, ge=-1, le=1, description="Correlation to S&P 500")
    correlation_dxy: float = Field(0.0, ge=-1, le=1, description="Correlation to US Dollar Index")


class LiquidityMetrics(BaseModel):
    """Liquidity metrics for a cryptocurrency on a specific exchange.

    Critical for assessing execution risk and market impact costs.
    """
    asset: str = Field(..., description="Asset identifier")
    exchange: str = Field(..., description="Exchange name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Spread metrics
    bid_ask_spread_bps: float = Field(..., ge=0, description="Bid-ask spread in basis points")

    # Depth metrics
    order_book_depth_usd: float = Field(..., ge=0, description="Total order book depth within 2% of mid (USD)")
    bid_depth_usd: float = Field(0.0, ge=0, description="Bid-side depth within 2% (USD)")
    ask_depth_usd: float = Field(0.0, ge=0, description="Ask-side depth within 2% (USD)")

    # Volume metrics
    daily_volume_usd: float = Field(..., ge=0, description="24-hour trading volume (USD)")
    avg_trade_size_usd: float = Field(0.0, ge=0, description="Average trade size (USD)")

    # Market impact estimates
    slippage_estimate_100k: float = Field(0.0, ge=0, description="Estimated slippage for $100K order (bps)")
    slippage_estimate_1m: float = Field(..., ge=0, description="Estimated slippage for $1M order (bps)")
    slippage_estimate_10m: float = Field(0.0, ge=0, description="Estimated slippage for $10M order (bps)")

    # Liquidity score
    liquidity_score: float = Field(0.0, ge=0, le=100, description="Composite liquidity score (0-100)")


class MarketRiskReport(BaseModel):
    """Comprehensive market risk report for a digital asset position.

    Combines volatility, liquidity, and position-specific risk metrics
    with actionable recommendations.
    """
    asset: str
    position_size_usd: float
    holding_period_days: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Risk metrics
    volatility: CryptoVolatilityMetrics
    liquidity: Optional[LiquidityMetrics] = None

    # Position-specific VaR
    var_95_usd: float = Field(..., description="95% VaR in USD for position")
    var_99_usd: float = Field(..., description="99% VaR in USD for position")
    cvar_95_usd: float = Field(..., description="95% Expected Shortfall in USD")

    # Composite scores
    risk_score: float = Field(..., ge=0, le=100, description="Composite risk score (0-100)")
    risk_rating: RiskRating

    # Risk factors and recommendations
    risk_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # Regulatory considerations
    regulatory_flags: list[str] = Field(default_factory=list)


# =============================================================================
# Router Request/Response Models
# =============================================================================


class RiskAssessmentRequest(BaseModel):
    """Request for position risk assessment."""

    asset: str = Field(..., description="Asset identifier (e.g., BTC, ETH, SOL)", examples=["BTC", "ETH"])
    position_size_usd: float = Field(..., gt=0, description="Position size in USD", examples=[1_000_000])
    holding_period_days: int = Field(1, ge=1, le=365, description="Expected holding period in days")
    confidence_level: float = Field(0.95, ge=0.90, le=0.99, description="Confidence level for VaR")
    include_liquidity: bool = Field(True, description="Include liquidity risk analysis")


class RiskAssessmentResponse(BaseModel):
    """Position risk assessment response."""

    asset: str
    position_size_usd: float
    holding_period_days: int

    # VaR metrics
    var_95_pct: float = Field(..., description="95% VaR as percentage")
    var_99_pct: float = Field(..., description="99% VaR as percentage")
    var_usd: float = Field(..., description="VaR in USD at requested confidence")
    cvar_usd: float = Field(..., description="Conditional VaR (Expected Shortfall) in USD")

    # Risk classification
    risk_rating: RiskRating
    risk_score: float = Field(..., ge=0, le=100)

    # Details
    volatility_30d: float = Field(..., description="30-day annualized volatility")
    max_drawdown: float = Field(..., description="Historical maximum drawdown")
    key_risks: list[str]
    recommendations: list[str]

    # Liquidity (optional)
    liquidity_score: Optional[float] = Field(None, description="Liquidity score 0-100")
    estimated_slippage_bps: Optional[float] = Field(None, description="Estimated slippage for position")


class MarketIntelligenceResponse(BaseModel):
    """Market intelligence for a digital asset."""

    asset: str
    timestamp: str

    # Price metrics
    current_price_usd: float
    price_change_24h_pct: float
    price_change_7d_pct: float
    price_change_30d_pct: float

    # Volatility
    volatility_30d: float
    volatility_90d: float

    # Correlations
    correlation_btc: float
    correlation_eth: float
    correlation_spy: float

    # Risk metrics
    var_95_1d: float
    max_drawdown: float
    risk_rating: RiskRating

    # Market structure
    daily_volume_usd: float
    liquidity_score: float

    # Regulatory context
    regulatory_notes: list[str] = Field(default_factory=list)


class VaRCalculationRequest(BaseModel):
    """Direct VaR calculation request."""

    volatility: float = Field(..., gt=0, le=5.0, description="Annualized volatility (e.g., 0.8 for 80%)")
    position_size_usd: float = Field(..., gt=0)
    confidence_level: float = Field(0.95, ge=0.90, le=0.99)
    holding_period_days: int = Field(1, ge=1, le=365)


class VaRCalculationResponse(BaseModel):
    """VaR calculation result."""

    var_pct: float
    var_usd: float
    cvar_pct: float
    cvar_usd: float
    inputs: dict

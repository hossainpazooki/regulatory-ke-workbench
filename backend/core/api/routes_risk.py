"""
Market risk analytics endpoints for digital assets.

Provides risk metrics, VaR calculations, and market intelligence
for cryptocurrency portfolio risk assessment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.analytics.market_risk import (
    RiskRating,
    CryptoVolatilityMetrics,
    LiquidityMetrics,
    MarketRiskReport,
    calculate_volatility_metrics,
    calculate_liquidity_metrics,
    generate_market_risk_report,
    calculate_var,
    calculate_cvar,
)

router = APIRouter(prefix="/risk", tags=["market-risk"])


# Demo price data for common crypto assets (for demonstration purposes)
DEMO_PRICE_DATA: dict[str, list[float]] = {
    "BTC": [42000, 43500, 42800, 44200, 45000, 44500, 43800, 45500, 46200, 45800,
            44900, 46500, 47200, 46800, 48000, 47500, 49000, 48500, 50000, 49500,
            51000, 50500, 52000, 51500, 53000, 52500, 51800, 53500, 54000, 53200],
    "ETH": [2200, 2280, 2240, 2320, 2400, 2360, 2300, 2420, 2500, 2460,
            2380, 2520, 2600, 2550, 2680, 2620, 2750, 2700, 2850, 2800,
            2920, 2880, 3000, 2950, 3100, 3050, 2980, 3150, 3200, 3120],
    "SOL": [95, 102, 98, 108, 115, 110, 105, 118, 125, 120,
            112, 128, 135, 130, 142, 138, 150, 145, 158, 152,
            165, 160, 172, 168, 180, 175, 170, 185, 190, 182],
    "USDC": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# Demo liquidity data
DEMO_LIQUIDITY: dict[str, dict] = {
    "BTC": {"spread_bps": 5, "depth_usd": 50_000_000, "volume_usd": 25_000_000_000},
    "ETH": {"spread_bps": 8, "depth_usd": 30_000_000, "volume_usd": 15_000_000_000},
    "SOL": {"spread_bps": 15, "depth_usd": 10_000_000, "volume_usd": 2_000_000_000},
    "USDC": {"spread_bps": 2, "depth_usd": 100_000_000, "volume_usd": 50_000_000_000},
}


class RiskAssessmentRequest(BaseModel):
    """Request for position risk assessment."""

    asset: str = Field(
        ...,
        description="Asset identifier (e.g., BTC, ETH, SOL)",
        examples=["BTC", "ETH"],
    )
    position_size_usd: float = Field(
        ...,
        gt=0,
        description="Position size in USD",
        examples=[1_000_000],
    )
    holding_period_days: int = Field(
        1,
        ge=1,
        le=365,
        description="Expected holding period in days",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.90,
        le=0.99,
        description="Confidence level for VaR calculation",
    )
    include_liquidity: bool = Field(
        True,
        description="Include liquidity risk analysis",
    )


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

    # Regulatory context (if requested)
    regulatory_notes: list[str] = Field(default_factory=list)


class VaRCalculationRequest(BaseModel):
    """Direct VaR calculation request."""

    volatility: float = Field(
        ...,
        gt=0,
        le=5.0,
        description="Annualized volatility (e.g., 0.8 for 80%)",
    )
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


@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_position_risk(request: RiskAssessmentRequest) -> RiskAssessmentResponse:
    """
    Assess market risk for a digital asset position.

    Calculates VaR, CVaR, and provides risk classification
    with recommendations.
    """
    asset = request.asset.upper()

    # Get price data (demo or real)
    price_data = DEMO_PRICE_DATA.get(asset)
    if not price_data:
        raise HTTPException(
            status_code=404,
            detail=f"Asset {asset} not found. Available: {list(DEMO_PRICE_DATA.keys())}",
        )

    # Calculate volatility metrics
    btc_prices = DEMO_PRICE_DATA.get("BTC", [])
    eth_prices = DEMO_PRICE_DATA.get("ETH", [])

    # Calculate returns for correlation
    def to_returns(prices: list[float]) -> list[float]:
        import math
        return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]

    btc_returns = to_returns(btc_prices) if btc_prices else None
    eth_returns = to_returns(eth_prices) if eth_prices else None

    volatility_metrics = calculate_volatility_metrics(
        asset_id=asset,
        price_history=price_data,
        btc_returns=btc_returns if asset != "BTC" else None,
        eth_returns=eth_returns if asset != "ETH" else None,
    )

    # Calculate liquidity metrics if requested
    liquidity_metrics = None
    if request.include_liquidity:
        liq_data = DEMO_LIQUIDITY.get(asset, DEMO_LIQUIDITY["SOL"])
        liquidity_metrics = calculate_liquidity_metrics(
            asset_id=asset,
            exchange="aggregated",
            bid_ask_spread_bps=liq_data["spread_bps"],
            order_book_depth_usd=liq_data["depth_usd"],
            daily_volume_usd=liq_data["volume_usd"],
        )

    # Generate full report
    report = generate_market_risk_report(
        asset_id=asset,
        position_size_usd=request.position_size_usd,
        volatility_metrics=volatility_metrics,
        liquidity_metrics=liquidity_metrics,
        holding_period_days=request.holding_period_days,
    )

    # Calculate VaR at requested confidence level
    var_at_conf = calculate_var(
        volatility_metrics.rolling_volatility_30d,
        request.confidence_level,
        request.holding_period_days,
    )
    cvar_at_conf = calculate_cvar(
        volatility_metrics.rolling_volatility_30d,
        request.confidence_level,
        request.holding_period_days,
    )

    return RiskAssessmentResponse(
        asset=asset,
        position_size_usd=request.position_size_usd,
        holding_period_days=request.holding_period_days,
        var_95_pct=volatility_metrics.var_95,
        var_99_pct=volatility_metrics.var_99,
        var_usd=request.position_size_usd * var_at_conf,
        cvar_usd=request.position_size_usd * cvar_at_conf,
        risk_rating=report.risk_rating,
        risk_score=report.risk_score,
        volatility_30d=volatility_metrics.rolling_volatility_30d,
        max_drawdown=volatility_metrics.max_drawdown,
        key_risks=report.risk_factors,
        recommendations=report.recommendations,
        liquidity_score=liquidity_metrics.liquidity_score if liquidity_metrics else None,
        estimated_slippage_bps=liquidity_metrics.slippage_estimate_1m if liquidity_metrics else None,
    )


@router.get("/market-intelligence/{asset}", response_model=MarketIntelligenceResponse)
async def get_market_intelligence(
    asset: str,
    include_regulatory: bool = Query(True, description="Include regulatory context"),
    include_correlations: bool = Query(True, description="Include correlation analysis"),
) -> MarketIntelligenceResponse:
    """
    Get market intelligence for a digital asset.

    Returns volatility metrics, correlations, and regulatory context.
    """
    asset = asset.upper()

    price_data = DEMO_PRICE_DATA.get(asset)
    if not price_data:
        raise HTTPException(
            status_code=404,
            detail=f"Asset {asset} not found. Available: {list(DEMO_PRICE_DATA.keys())}",
        )

    # Calculate volatility
    def to_returns(prices: list[float]) -> list[float]:
        import math
        return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]

    btc_returns = to_returns(DEMO_PRICE_DATA.get("BTC", [])) if include_correlations else None
    eth_returns = to_returns(DEMO_PRICE_DATA.get("ETH", [])) if include_correlations else None

    volatility_metrics = calculate_volatility_metrics(
        asset_id=asset,
        price_history=price_data,
        btc_returns=btc_returns if asset != "BTC" else None,
        eth_returns=eth_returns if asset != "ETH" else None,
    )

    # Get liquidity
    liq_data = DEMO_LIQUIDITY.get(asset, DEMO_LIQUIDITY["SOL"])
    liquidity_metrics = calculate_liquidity_metrics(
        asset_id=asset,
        exchange="aggregated",
        bid_ask_spread_bps=liq_data["spread_bps"],
        order_book_depth_usd=liq_data["depth_usd"],
        daily_volume_usd=liq_data["volume_usd"],
    )

    # Price changes
    current_price = price_data[-1]
    change_24h = (price_data[-1] - price_data[-2]) / price_data[-2] if len(price_data) >= 2 else 0
    change_7d = (price_data[-1] - price_data[-7]) / price_data[-7] if len(price_data) >= 7 else 0
    change_30d = (price_data[-1] - price_data[0]) / price_data[0] if len(price_data) >= 30 else 0

    # Determine risk rating
    if volatility_metrics.rolling_volatility_30d > 1.0:
        risk_rating = RiskRating.EXTREME
    elif volatility_metrics.rolling_volatility_30d > 0.6:
        risk_rating = RiskRating.HIGH
    elif volatility_metrics.rolling_volatility_30d > 0.3:
        risk_rating = RiskRating.MEDIUM
    else:
        risk_rating = RiskRating.LOW

    # Regulatory notes
    regulatory_notes = []
    if include_regulatory:
        if asset in ["BTC", "ETH"]:
            regulatory_notes.append("Commodity treatment likely under CFTC jurisdiction")
        if asset == "USDC":
            regulatory_notes.append("Payment stablecoin - GENIUS Act compliance required")
            regulatory_notes.append("Reserve attestation requirements apply")
        if volatility_metrics.rolling_volatility_30d > 0.8:
            regulatory_notes.append("High volatility may trigger enhanced risk disclosures")

    return MarketIntelligenceResponse(
        asset=asset,
        timestamp=datetime.now(timezone.utc).isoformat(),
        current_price_usd=current_price,
        price_change_24h_pct=change_24h,
        price_change_7d_pct=change_7d,
        price_change_30d_pct=change_30d,
        volatility_30d=volatility_metrics.rolling_volatility_30d,
        volatility_90d=volatility_metrics.rolling_volatility_90d,
        correlation_btc=volatility_metrics.correlation_btc,
        correlation_eth=volatility_metrics.correlation_eth,
        correlation_spy=volatility_metrics.correlation_spy,
        var_95_1d=volatility_metrics.var_95,
        max_drawdown=volatility_metrics.max_drawdown,
        risk_rating=risk_rating,
        daily_volume_usd=liq_data["volume_usd"],
        liquidity_score=liquidity_metrics.liquidity_score,
        regulatory_notes=regulatory_notes,
    )


@router.post("/calculate-var", response_model=VaRCalculationResponse)
async def calculate_var_endpoint(request: VaRCalculationRequest) -> VaRCalculationResponse:
    """
    Calculate Value at Risk and Conditional VaR directly.

    Useful for custom volatility inputs or scenario analysis.
    """
    var_pct = calculate_var(
        request.volatility,
        request.confidence_level,
        request.holding_period_days,
    )
    cvar_pct = calculate_cvar(
        request.volatility,
        request.confidence_level,
        request.holding_period_days,
    )

    return VaRCalculationResponse(
        var_pct=var_pct,
        var_usd=request.position_size_usd * var_pct,
        cvar_pct=cvar_pct,
        cvar_usd=request.position_size_usd * cvar_pct,
        inputs={
            "volatility": request.volatility,
            "position_size_usd": request.position_size_usd,
            "confidence_level": request.confidence_level,
            "holding_period_days": request.holding_period_days,
        },
    )


@router.get("/supported-assets")
async def list_supported_assets() -> dict:
    """List all supported assets for risk analysis."""
    return {
        "assets": list(DEMO_PRICE_DATA.keys()),
        "note": "Demo data used for demonstration. Production would integrate with market data providers.",
    }

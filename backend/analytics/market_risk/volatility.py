"""
Market risk metrics for cryptocurrency and digital assets.

Implements Value at Risk (VaR), Conditional VaR (Expected Shortfall),
volatility analysis, and liquidity metrics for crypto portfolios.

References:
- Basel III market risk framework
- FRTB standardized approach for crypto assets
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskRating(str, Enum):
    """Risk rating classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class CryptoVolatilityMetrics(BaseModel):
    """
    Volatility metrics for a cryptocurrency asset.

    Includes historical volatility, VaR, CVaR, and correlation metrics
    essential for portfolio risk management.
    """
    asset: str = Field(..., description="Asset identifier (e.g., BTC, ETH)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Volatility measures
    rolling_volatility_30d: float = Field(
        ...,
        ge=0,
        description="30-day rolling annualized volatility"
    )
    rolling_volatility_90d: float = Field(
        0.0,
        ge=0,
        description="90-day rolling annualized volatility"
    )

    # Value at Risk (parametric, assuming normal distribution)
    var_95: float = Field(
        ...,
        description="95% 1-day Value at Risk (percentage)"
    )
    var_99: float = Field(
        ...,
        description="99% 1-day Value at Risk (percentage)"
    )

    # Conditional VaR (Expected Shortfall)
    cvar_95: float = Field(
        ...,
        description="95% Expected Shortfall (percentage)"
    )
    cvar_99: float = Field(
        0.0,
        description="99% Expected Shortfall (percentage)"
    )

    # Drawdown metrics
    max_drawdown: float = Field(
        ...,
        ge=0,
        le=1,
        description="Maximum drawdown observed (0-1)"
    )
    current_drawdown: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Current drawdown from recent high"
    )

    # Correlation to major assets (for portfolio diversification analysis)
    correlation_btc: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Correlation to Bitcoin"
    )
    correlation_eth: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Correlation to Ethereum"
    )
    correlation_spy: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Correlation to S&P 500"
    )
    correlation_dxy: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Correlation to US Dollar Index"
    )


class LiquidityMetrics(BaseModel):
    """
    Liquidity metrics for a cryptocurrency on a specific exchange.

    Critical for assessing execution risk and market impact costs.
    """
    asset: str = Field(..., description="Asset identifier")
    exchange: str = Field(..., description="Exchange name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Spread metrics
    bid_ask_spread_bps: float = Field(
        ...,
        ge=0,
        description="Bid-ask spread in basis points"
    )

    # Depth metrics
    order_book_depth_usd: float = Field(
        ...,
        ge=0,
        description="Total order book depth within 2% of mid (USD)"
    )
    bid_depth_usd: float = Field(
        0.0,
        ge=0,
        description="Bid-side depth within 2% (USD)"
    )
    ask_depth_usd: float = Field(
        0.0,
        ge=0,
        description="Ask-side depth within 2% (USD)"
    )

    # Volume metrics
    daily_volume_usd: float = Field(
        ...,
        ge=0,
        description="24-hour trading volume (USD)"
    )
    avg_trade_size_usd: float = Field(
        0.0,
        ge=0,
        description="Average trade size (USD)"
    )

    # Market impact estimates
    slippage_estimate_100k: float = Field(
        0.0,
        ge=0,
        description="Estimated slippage for $100K order (bps)"
    )
    slippage_estimate_1m: float = Field(
        ...,
        ge=0,
        description="Estimated slippage for $1M order (bps)"
    )
    slippage_estimate_10m: float = Field(
        0.0,
        ge=0,
        description="Estimated slippage for $10M order (bps)"
    )

    # Liquidity score
    liquidity_score: float = Field(
        0.0,
        ge=0,
        le=100,
        description="Composite liquidity score (0-100, higher is better)"
    )


class MarketRiskReport(BaseModel):
    """
    Comprehensive market risk report for a digital asset position.

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
    var_95_usd: float = Field(
        ...,
        description="95% VaR in USD for position"
    )
    var_99_usd: float = Field(
        ...,
        description="99% VaR in USD for position"
    )
    cvar_95_usd: float = Field(
        ...,
        description="95% Expected Shortfall in USD"
    )

    # Composite scores
    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite risk score (0-100, higher is riskier)"
    )
    risk_rating: RiskRating

    # Risk factors and recommendations
    risk_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # Regulatory considerations
    regulatory_flags: list[str] = Field(default_factory=list)


# Standard normal distribution z-scores
Z_SCORES = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
    0.995: 2.576,
}

# CVaR multipliers (for normal distribution)
CVAR_MULTIPLIERS = {
    0.95: 2.063,  # E[X | X > z_0.95] for standard normal
    0.99: 2.665,
}


def calculate_var(
    volatility: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
) -> float:
    """
    Calculate parametric Value at Risk.

    Uses the variance-covariance method assuming normal returns.

    Args:
        volatility: Annualized volatility (e.g., 0.80 for 80%)
        confidence_level: Confidence level (0.90, 0.95, 0.99)
        holding_period_days: Holding period in days

    Returns:
        VaR as percentage of position value
    """
    z_score = Z_SCORES.get(confidence_level, 1.645)

    # Scale volatility to holding period (assuming 365 trading days for crypto)
    daily_vol = volatility / math.sqrt(365)
    period_vol = daily_vol * math.sqrt(holding_period_days)

    return z_score * period_vol


def calculate_cvar(
    volatility: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR represents the expected loss given that VaR is exceeded.
    More conservative than VaR for risk management.

    Args:
        volatility: Annualized volatility
        confidence_level: Confidence level
        holding_period_days: Holding period in days

    Returns:
        CVaR as percentage of position value
    """
    multiplier = CVAR_MULTIPLIERS.get(confidence_level, 2.063)

    daily_vol = volatility / math.sqrt(365)
    period_vol = daily_vol * math.sqrt(holding_period_days)

    return multiplier * period_vol


def calculate_volatility_metrics(
    asset_id: str,
    price_history: list[float],
    btc_returns: Optional[list[float]] = None,
    eth_returns: Optional[list[float]] = None,
    spy_returns: Optional[list[float]] = None,
) -> CryptoVolatilityMetrics:
    """
    Calculate comprehensive volatility metrics from price history.

    Args:
        asset_id: Asset identifier
        price_history: List of historical prices (most recent last)
        btc_returns: Optional BTC returns for correlation
        eth_returns: Optional ETH returns for correlation
        spy_returns: Optional S&P 500 returns for correlation

    Returns:
        CryptoVolatilityMetrics with calculated values
    """
    if len(price_history) < 2:
        raise ValueError("Need at least 2 price points")

    # Calculate log returns
    returns = []
    for i in range(1, len(price_history)):
        if price_history[i-1] > 0:
            ret = math.log(price_history[i] / price_history[i-1])
            returns.append(ret)

    if not returns:
        raise ValueError("Could not calculate returns from prices")

    # Calculate rolling volatility (annualized)
    def calc_vol(ret_slice: list[float]) -> float:
        if len(ret_slice) < 2:
            return 0.0
        mean = sum(ret_slice) / len(ret_slice)
        variance = sum((r - mean) ** 2 for r in ret_slice) / (len(ret_slice) - 1)
        daily_vol = math.sqrt(variance)
        return daily_vol * math.sqrt(365)  # Annualize for crypto (365 days)

    # Use available data for 30d and 90d volatility
    vol_30d = calc_vol(returns[-30:]) if len(returns) >= 30 else calc_vol(returns)
    vol_90d = calc_vol(returns[-90:]) if len(returns) >= 90 else calc_vol(returns)

    # Calculate VaR and CVaR
    var_95 = calculate_var(vol_30d, 0.95, 1)
    var_99 = calculate_var(vol_30d, 0.99, 1)
    cvar_95 = calculate_cvar(vol_30d, 0.95, 1)
    cvar_99 = calculate_cvar(vol_30d, 0.99, 1)

    # Calculate max drawdown
    peak = price_history[0]
    max_dd = 0.0
    current_dd = 0.0
    for price in price_history:
        if price > peak:
            peak = price
        dd = (peak - price) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        current_dd = dd

    # Calculate correlations if comparison returns provided
    def calc_correlation(returns_a: list[float], returns_b: list[float]) -> float:
        n = min(len(returns_a), len(returns_b))
        if n < 10:
            return 0.0
        a = returns_a[-n:]
        b = returns_b[-n:]
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / (n - 1)
        std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a) / (n - 1))
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b) / (n - 1))
        if std_a == 0 or std_b == 0:
            return 0.0
        return cov / (std_a * std_b)

    corr_btc = calc_correlation(returns, btc_returns) if btc_returns else 0.0
    corr_eth = calc_correlation(returns, eth_returns) if eth_returns else 0.0
    corr_spy = calc_correlation(returns, spy_returns) if spy_returns else 0.0

    return CryptoVolatilityMetrics(
        asset=asset_id,
        rolling_volatility_30d=vol_30d,
        rolling_volatility_90d=vol_90d,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        max_drawdown=max_dd,
        current_drawdown=current_dd,
        correlation_btc=corr_btc,
        correlation_eth=corr_eth,
        correlation_spy=corr_spy,
    )


def calculate_liquidity_metrics(
    asset_id: str,
    exchange: str,
    bid_ask_spread_bps: float,
    order_book_depth_usd: float,
    daily_volume_usd: float,
    bid_depth_usd: Optional[float] = None,
    ask_depth_usd: Optional[float] = None,
) -> LiquidityMetrics:
    """
    Calculate liquidity metrics and market impact estimates.

    Args:
        asset_id: Asset identifier
        exchange: Exchange name
        bid_ask_spread_bps: Bid-ask spread in basis points
        order_book_depth_usd: Total order book depth (USD)
        daily_volume_usd: 24h trading volume (USD)
        bid_depth_usd: Optional bid-side depth
        ask_depth_usd: Optional ask-side depth

    Returns:
        LiquidityMetrics with slippage estimates
    """
    # Estimate slippage using square-root market impact model
    # Slippage = k * sqrt(order_size / daily_volume)
    # k is calibrated based on empirical crypto market data
    k = 0.1  # Market impact coefficient

    def estimate_slippage(order_size: float) -> float:
        if daily_volume_usd <= 0:
            return 10000  # 100% slippage if no volume
        impact = k * math.sqrt(order_size / daily_volume_usd)
        # Add half spread
        return (impact * 10000) + (bid_ask_spread_bps / 2)

    slippage_100k = estimate_slippage(100_000)
    slippage_1m = estimate_slippage(1_000_000)
    slippage_10m = estimate_slippage(10_000_000)

    # Calculate composite liquidity score
    # Higher volume, tighter spread, deeper book = higher score
    volume_score = min(50, 50 * math.log10(max(1, daily_volume_usd / 1_000_000) + 1) / 3)
    spread_score = max(0, 25 - bid_ask_spread_bps / 4)
    depth_score = min(25, 25 * math.log10(max(1, order_book_depth_usd / 100_000) + 1) / 2)
    liquidity_score = volume_score + spread_score + depth_score

    return LiquidityMetrics(
        asset=asset_id,
        exchange=exchange,
        bid_ask_spread_bps=bid_ask_spread_bps,
        order_book_depth_usd=order_book_depth_usd,
        bid_depth_usd=bid_depth_usd or order_book_depth_usd / 2,
        ask_depth_usd=ask_depth_usd or order_book_depth_usd / 2,
        daily_volume_usd=daily_volume_usd,
        slippage_estimate_100k=slippage_100k,
        slippage_estimate_1m=slippage_1m,
        slippage_estimate_10m=slippage_10m,
        liquidity_score=liquidity_score,
    )


def generate_market_risk_report(
    asset_id: str,
    position_size_usd: float,
    volatility_metrics: CryptoVolatilityMetrics,
    liquidity_metrics: Optional[LiquidityMetrics] = None,
    holding_period_days: int = 1,
) -> MarketRiskReport:
    """
    Generate comprehensive market risk report for a position.

    Args:
        asset_id: Asset identifier
        position_size_usd: Position size in USD
        volatility_metrics: Pre-calculated volatility metrics
        liquidity_metrics: Optional liquidity metrics
        holding_period_days: Expected holding period

    Returns:
        MarketRiskReport with risk analysis and recommendations
    """
    risk_factors = []
    recommendations = []
    regulatory_flags = []

    # Scale VaR to holding period
    var_95 = calculate_var(
        volatility_metrics.rolling_volatility_30d, 0.95, holding_period_days
    )
    var_99 = calculate_var(
        volatility_metrics.rolling_volatility_30d, 0.99, holding_period_days
    )
    cvar_95 = calculate_cvar(
        volatility_metrics.rolling_volatility_30d, 0.95, holding_period_days
    )

    # Position-specific VaR in USD
    var_95_usd = position_size_usd * var_95
    var_99_usd = position_size_usd * var_99
    cvar_95_usd = position_size_usd * cvar_95

    # Identify risk factors
    if volatility_metrics.rolling_volatility_30d > 1.0:  # >100% annualized
        risk_factors.append("Extreme volatility (>100% annualized)")
    elif volatility_metrics.rolling_volatility_30d > 0.6:
        risk_factors.append("High volatility (>60% annualized)")

    if volatility_metrics.max_drawdown > 0.5:
        risk_factors.append(f"Historical max drawdown of {volatility_metrics.max_drawdown:.0%}")

    if volatility_metrics.current_drawdown > 0.2:
        risk_factors.append(f"Currently in {volatility_metrics.current_drawdown:.0%} drawdown")

    if volatility_metrics.correlation_btc > 0.8:
        risk_factors.append("High BTC correlation - limited diversification benefit")

    if volatility_metrics.correlation_spy < -0.3:
        risk_factors.append("Negative equity correlation - potential hedge properties")

    # Liquidity risk factors
    if liquidity_metrics:
        if liquidity_metrics.slippage_estimate_1m > 100:  # >1% slippage
            risk_factors.append("High market impact for institutional sizes")
        if liquidity_metrics.daily_volume_usd < position_size_usd * 10:
            risk_factors.append("Position exceeds 10% of daily volume")
            recommendations.append("Consider executing over multiple days")
        if liquidity_metrics.bid_ask_spread_bps > 50:
            risk_factors.append("Wide bid-ask spread (>50 bps)")

    # Calculate composite risk score (0-100)
    vol_component = min(40, volatility_metrics.rolling_volatility_30d * 40)
    dd_component = volatility_metrics.max_drawdown * 30

    liquidity_penalty = 0
    if liquidity_metrics:
        if liquidity_metrics.liquidity_score < 50:
            liquidity_penalty = (50 - liquidity_metrics.liquidity_score) * 0.6

    risk_score = vol_component + dd_component + liquidity_penalty
    risk_score = min(100, max(0, risk_score))

    # Determine risk rating
    if risk_score >= 75:
        risk_rating = RiskRating.EXTREME
        recommendations.append("Consider reducing position size")
        recommendations.append("Implement stop-loss orders")
    elif risk_score >= 50:
        risk_rating = RiskRating.HIGH
        recommendations.append("Monitor volatility daily")
        recommendations.append("Consider hedging strategies")
    elif risk_score >= 25:
        risk_rating = RiskRating.MEDIUM
        recommendations.append("Regular portfolio rebalancing recommended")
    else:
        risk_rating = RiskRating.LOW

    # Regulatory flags
    if position_size_usd > 10_000_000:
        regulatory_flags.append("Large position - may trigger reporting requirements")
    if var_99_usd > 1_000_000:
        regulatory_flags.append("Material VaR exposure - document risk management controls")

    return MarketRiskReport(
        asset=asset_id,
        position_size_usd=position_size_usd,
        holding_period_days=holding_period_days,
        volatility=volatility_metrics,
        liquidity=liquidity_metrics,
        var_95_usd=var_95_usd,
        var_99_usd=var_99_usd,
        cvar_95_usd=cvar_95_usd,
        risk_score=risk_score,
        risk_rating=risk_rating,
        risk_factors=risk_factors,
        recommendations=recommendations,
        regulatory_flags=regulatory_flags,
    )

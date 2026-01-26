"""
DeFi risk scoring schemas.

Pydantic models for DeFi protocol risk assessment across:
- Smart contract risk
- Economic risk
- Oracle risk
- Governance risk
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DeFiCategory(str, Enum):
    """DeFi protocol categories."""

    STAKING = "staking"
    LIQUIDITY_POOL = "liquidity_pool"
    LENDING = "lending"
    BRIDGE = "bridge"
    DEX = "dex"
    YIELD_AGGREGATOR = "yield_aggregator"
    DERIVATIVES = "derivatives"
    STABLECOIN = "stablecoin"
    INSURANCE = "insurance"
    RESTAKING = "restaking"


class GovernanceType(str, Enum):
    """Governance mechanism types."""

    TOKEN_VOTING = "token_voting"
    MULTISIG = "multisig"
    OPTIMISTIC = "optimistic"
    IMMUTABLE = "immutable"
    CENTRALIZED = "centralized"
    HYBRID = "hybrid"


class OracleProvider(str, Enum):
    """Oracle service providers."""

    CHAINLINK = "chainlink"
    PYTH = "pyth"
    BAND = "band"
    UNISWAP_TWAP = "uniswap_twap"
    CUSTOM = "custom"
    NONE = "none"


class RiskGrade(str, Enum):
    """Letter grade risk rating (A=lowest risk, F=highest risk)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class SmartContractRisk(BaseModel):
    """Smart contract risk assessment inputs."""

    audit_count: int = Field(0, ge=0, description="Number of independent security audits")
    auditors: list[str] = Field(default_factory=list, description="Names of audit firms")
    last_audit_days_ago: int = Field(365, ge=0, description="Days since last security audit")
    formal_verification: bool = Field(False, description="Whether formal verification was performed")
    is_upgradeable: bool = Field(True, description="Whether contracts are upgradeable")
    upgrade_timelock_hours: int = Field(0, ge=0, description="Timelock delay for upgrades in hours")
    has_admin_functions: bool = Field(True, description="Whether admin-only functions exist")
    admin_can_pause: bool = Field(True, description="Whether admin can pause the protocol")
    admin_can_drain: bool = Field(False, description="Whether admin can withdraw user funds")
    tvl_usd: float = Field(0, ge=0, description="Total Value Locked in USD")
    contract_age_days: int = Field(0, ge=0, description="Days since mainnet deployment")
    exploit_history_count: int = Field(0, ge=0, description="Number of historical exploits")
    total_exploit_loss_usd: float = Field(0, ge=0, description="Total USD lost to exploits")
    bug_bounty_max_usd: float = Field(0, ge=0, description="Maximum bug bounty payout")


class EconomicRisk(BaseModel):
    """Economic and tokenomics risk assessment."""

    token_concentration_top10_pct: float = Field(50.0, ge=0, le=100)
    team_token_pct: float = Field(20.0, ge=0, le=100)
    vesting_remaining_pct: float = Field(50.0, ge=0, le=100)
    treasury_runway_months: float = Field(24.0, ge=0)
    treasury_diversified: bool = Field(False)
    has_protocol_revenue: bool = Field(True)
    revenue_30d_usd: float = Field(0, ge=0)
    has_impermanent_loss: bool = Field(False)
    has_liquidation_risk: bool = Field(False)
    max_leverage: float = Field(1.0, ge=1.0)


class OracleRisk(BaseModel):
    """Oracle dependency risk assessment."""

    primary_oracle: OracleProvider = Field(OracleProvider.CHAINLINK)
    has_fallback_oracle: bool = Field(False)
    oracle_update_frequency_seconds: int = Field(3600, ge=1)
    oracle_manipulation_resistant: bool = Field(True)
    oracle_decentralized: bool = Field(True)
    oracle_failure_count_12m: int = Field(0, ge=0)
    oracle_deviation_threshold_pct: float = Field(1.0, gt=0)


class GovernanceRisk(BaseModel):
    """Governance and centralization risk assessment."""

    governance_type: GovernanceType = Field(GovernanceType.TOKEN_VOTING)
    has_timelock: bool = Field(True)
    timelock_hours: int = Field(48, ge=0)
    multisig_threshold: Optional[str] = Field(None)
    multisig_signers_doxxed: bool = Field(False)
    governance_participation_pct: float = Field(10.0, ge=0, le=100)
    quorum_pct: float = Field(4.0, ge=0, le=100)
    has_emergency_admin: bool = Field(True)
    emergency_actions_12m: int = Field(0, ge=0)


class DeFiRiskScore(BaseModel):
    """Comprehensive DeFi protocol risk score."""

    protocol_id: str
    category: DeFiCategory
    smart_contract_grade: RiskGrade
    economic_grade: RiskGrade
    oracle_grade: RiskGrade
    governance_grade: RiskGrade
    overall_grade: RiskGrade
    overall_score: float = Field(..., ge=0, le=100)
    smart_contract_score: float
    economic_score: float
    oracle_score: float
    governance_score: float
    critical_risks: list[str] = Field(default_factory=list)
    high_risks: list[str] = Field(default_factory=list)
    medium_risks: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    regulatory_flags: list[str] = Field(default_factory=list)
    metrics_summary: dict = Field(default_factory=dict)


# Request/Response schemas for API
class DeFiScoreRequest(BaseModel):
    """Request model for scoring a DeFi protocol."""

    protocol_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Protocol identifier (alphanumeric, underscores, hyphens only)",
    )
    category: DeFiCategory
    smart_contract: SmartContractRisk = Field(default_factory=SmartContractRisk)
    economic: EconomicRisk = Field(default_factory=EconomicRisk)
    oracle: OracleRisk = Field(default_factory=OracleRisk)
    governance: GovernanceRisk = Field(default_factory=GovernanceRisk)

# Market Risk Modules

Technical documentation for the market risk assessment capabilities in the regulatory workbench.

## Overview

The market risk modules provide quantitative risk assessment for:
- **Token Standards** - US regulatory compliance analysis (SEC Howey Test, GENIUS Act)
- **Protocol Risk** - Blockchain protocol risk scoring (consensus, decentralization, settlement)
- **DeFi Risk** - DeFi protocol risk grading (smart contract, economic, oracle, governance)
- **Market Analytics** - VaR, CVaR, and liquidity metrics

---

## Module Structure

```
backend/rule_service/app/services/
├── token_standards/
│   ├── __init__.py
│   └── compliance.py      # SEC Howey Test, GENIUS Act analysis
├── protocol_risk/
│   ├── __init__.py
│   └── consensus.py       # Blockchain protocol assessment
├── defi_risk/
│   ├── __init__.py
│   └── scoring.py         # DeFi protocol scoring
└── market_risk/
    ├── __init__.py
    └── volatility.py      # VaR, CVaR, liquidity metrics
```

---

## 1. Token Standards Module

**Location:** `rule_service/app/services/token_standards/compliance.py`

### Enums

| Enum | Values |
|------|--------|
| `TokenStandard` | ERC_20, ERC_721, ERC_1155, BEP_20, SPL, TRC_20 |
| `TokenClassification` | PAYMENT_STABLECOIN, SECURITY_TOKEN, UTILITY_TOKEN, COMMODITY_TOKEN, HYBRID_TOKEN, UNCLASSIFIED |

### Models

#### `HoweyTestResult`
SEC Howey Test 4-prong analysis result.

```python
class HoweyTestResult(BaseModel):
    is_security: bool
    investment_of_money: bool
    common_enterprise: bool
    expectation_of_profit: bool
    efforts_of_others: bool
    prong_scores: dict[str, float]
    analysis_notes: list[str]
```

#### `GeniusActAnalysis`
GENIUS Act stablecoin compliance analysis.

```python
class GeniusActAnalysis(BaseModel):
    is_compliant_stablecoin: bool
    reserve_requirements_met: bool
    issuer_requirements: list[str]
    compliance_gaps: list[str]
    required_disclosures: list[str]
```

#### `TokenComplianceResult`
Comprehensive compliance result.

```python
class TokenComplianceResult(BaseModel):
    standard: TokenStandard
    classification: TokenClassification
    requires_sec_registration: bool
    genius_act_applicable: bool
    howey_analysis: Optional[HoweyTestResult]
    genius_analysis: Optional[GeniusActAnalysis]
    compliance_requirements: list[str]
    regulatory_risks: list[str]
    recommended_actions: list[str]
    sec_jurisdiction: bool
    cftc_jurisdiction: bool
```

### Functions

#### `apply_howey_test()`
```python
def apply_howey_test(
    *,
    investment_of_money: bool,
    common_enterprise: bool,
    expectation_of_profits: bool,
    efforts_of_others: bool,
    decentralization_score: float = 0.0,
    is_functional_network: bool = False,
) -> HoweyTestResult
```

#### `analyze_genius_act_compliance()`
```python
def analyze_genius_act_compliance(
    *,
    is_stablecoin: bool,
    pegged_currency: str,
    reserve_assets: list[str],
    reserve_ratio: float,
    uses_algorithmic_mechanism: bool,
    issuer_charter_type: str,
    has_reserve_attestation: bool,
    attestation_frequency_days: int,
) -> GeniusActAnalysis
```

#### `analyze_token_compliance()`
Main entry point for comprehensive token compliance analysis.

```python
def analyze_token_compliance(
    *,
    standard: TokenStandard,
    has_profit_expectation: bool,
    is_decentralized: bool,
    backed_by_fiat: bool,
    # Howey test inputs
    investment_of_money: bool = True,
    common_enterprise: bool = True,
    efforts_of_promoter: bool = True,
    decentralization_score: float = 0.0,
    is_functional_network: bool = False,
    # GENIUS Act inputs
    is_stablecoin: bool = False,
    pegged_currency: str = "USD",
    reserve_assets: Optional[list[str]] = None,
    reserve_ratio: float = 1.0,
    uses_algorithmic_mechanism: bool = False,
    issuer_charter_type: str = "non_bank_qualified",
    has_reserve_attestation: bool = False,
    attestation_frequency_days: int = 30,
) -> TokenComplianceResult
```

### Usage Example

```python
from backend.rule_service.app.services.token_standards import (
    analyze_token_compliance,
    TokenStandard,
)

result = analyze_token_compliance(
    standard=TokenStandard.ERC_20,
    has_profit_expectation=False,
    is_decentralized=True,
    backed_by_fiat=True,
    is_stablecoin=True,
    pegged_currency="USD",
    reserve_ratio=1.0,
    has_reserve_attestation=True,
)

print(f"Classification: {result.classification.value}")
print(f"SEC Registration Required: {result.requires_sec_registration}")
print(f"GENIUS Act Applicable: {result.genius_act_applicable}")
```

---

## 2. Protocol Risk Module

**Location:** `rule_service/app/services/protocol_risk/consensus.py`

### Enums

| Enum | Values |
|------|--------|
| `ConsensusMechanism` | POW, POS, DPOS, POA, POH, PBFT, HYBRID |
| `SettlementFinality` | PROBABILISTIC, DETERMINISTIC, ECONOMIC |
| `RiskTier` | TIER_1 (institutional), TIER_2 (established), TIER_3 (emerging), TIER_4 (high risk) |

### Models

#### `ProtocolRiskProfile`
Input metrics for protocol assessment.

```python
class ProtocolRiskProfile(BaseModel):
    protocol_id: str
    consensus: ConsensusMechanism
    finality_type: SettlementFinality
    validator_count: int
    nakamoto_coefficient: int
    top_10_stake_pct: float
    finality_time_seconds: float
    tps_average: float
    tps_peak: float
    uptime_30d_pct: float
    major_incidents_12m: int
    has_bug_bounty: bool
    audit_count: int
    time_since_last_upgrade_days: int
    total_staked_usd: Optional[float]
    slashing_enabled: bool
```

#### `ProtocolRiskAssessment`
Assessment result with dimension scores.

```python
class ProtocolRiskAssessment(BaseModel):
    protocol_id: str
    risk_tier: RiskTier
    # Dimension scores (0-100)
    consensus_score: float
    decentralization_score: float
    settlement_score: float
    operational_score: float
    security_score: float
    overall_score: float
    # Analysis
    risk_factors: list[str]
    strengths: list[str]
    regulatory_notes: list[str]
    metrics_summary: dict
```

### Scoring Weights

| Dimension | Weight |
|-----------|--------|
| Consensus | 25% |
| Decentralization | 20% |
| Settlement | 20% |
| Operational | 20% |
| Security | 15% |

### Pre-configured Protocols

`PROTOCOL_DEFAULTS` contains configurations for:
- Bitcoin (POW, Tier 1)
- Ethereum (POS, Tier 1)
- Solana (POH, Tier 2)
- Polygon (POS, Tier 2)
- Avalanche (POS, Tier 2)
- BNB Chain (POS, Tier 2)
- Tron (DPOS, Tier 3)

### Functions

#### `assess_protocol_risk()`
```python
def assess_protocol_risk(
    protocol_id: str,
    consensus: ConsensusMechanism,
    finality_type: SettlementFinality,
    validator_count: int,
    nakamoto_coefficient: int,
    finality_time_seconds: float,
    tps_average: float,
    tps_peak: float,
    uptime_30d_pct: float = 99.9,
    major_incidents_12m: int = 0,
    has_bug_bounty: bool = False,
    audit_count: int = 0,
    time_since_last_upgrade_days: int = 90,
    top_10_stake_pct: float = 0.0,
    total_staked_usd: Optional[float] = None,
    slashing_enabled: bool = False,
) -> ProtocolRiskAssessment
```

#### `get_protocol_defaults()`
```python
def get_protocol_defaults(protocol_id: str) -> Optional[dict]
```

### Usage Example

```python
from backend.rule_service.app.services.protocol_risk import (
    assess_protocol_risk,
    get_protocol_defaults,
    PROTOCOL_DEFAULTS,
)

# Using defaults
defaults = get_protocol_defaults("ethereum")
assessment = assess_protocol_risk(protocol_id="ethereum", **defaults)

print(f"Risk Tier: {assessment.risk_tier.value}")
print(f"Overall Score: {assessment.overall_score:.1f}")
print(f"Consensus Score: {assessment.consensus_score:.1f}")
print(f"Decentralization Score: {assessment.decentralization_score:.1f}")
```

---

## 3. DeFi Risk Module

**Location:** `rule_service/app/services/defi_risk/scoring.py`

### Enums

| Enum | Values |
|------|--------|
| `DeFiCategory` | STAKING, LIQUIDITY_POOL, LENDING, BRIDGE, DEX, YIELD_AGGREGATOR, DERIVATIVES, STABLECOIN, INSURANCE, RESTAKING |
| `GovernanceType` | MULTISIG, TOKEN_VOTING, OPTIMISTIC, CENTRALIZED, HYBRID |
| `OracleProvider` | CHAINLINK, PYTH, BAND, UMA, CUSTOM, NONE |
| `RiskGrade` | A (excellent), B (good), C (fair), D (poor), F (fail) |

### Input Models

#### `SmartContractRisk`
```python
class SmartContractRisk(BaseModel):
    audit_count: int
    auditors: list[str]
    last_audit_days_ago: int
    formal_verification: bool
    is_upgradeable: bool
    upgrade_timelock_hours: int
    has_admin_functions: bool
    admin_can_pause: bool
    admin_can_drain: bool
    tvl_usd: float
    contract_age_days: int
    exploit_history_count: int
    bug_bounty_max_usd: float
```

#### `EconomicRisk`
```python
class EconomicRisk(BaseModel):
    token_concentration_top10_pct: float
    team_token_pct: float
    treasury_runway_months: int
    treasury_diversified: bool
    has_protocol_revenue: bool
    revenue_30d_usd: float
    has_liquidation_risk: bool
    max_leverage: float
```

#### `OracleRisk`
```python
class OracleRisk(BaseModel):
    primary_oracle: OracleProvider
    has_fallback_oracle: bool
    oracle_update_frequency_seconds: int
    oracle_deviation_threshold_pct: float
    historical_oracle_failures: int
```

#### `GovernanceRisk`
```python
class GovernanceRisk(BaseModel):
    governance_type: GovernanceType
    has_timelock: bool
    timelock_delay_hours: int
    multisig_threshold: str  # e.g., "3/5"
    token_voting_quorum_pct: float
    proposal_delay_days: int
    can_upgrade_without_vote: bool
```

### Output Model

#### `DeFiRiskScore`
```python
class DeFiRiskScore(BaseModel):
    protocol_id: str
    category: DeFiCategory
    # Grades
    smart_contract_grade: RiskGrade
    economic_grade: RiskGrade
    oracle_grade: RiskGrade
    governance_grade: RiskGrade
    overall_grade: RiskGrade
    overall_score: float
    # Scores (0-100)
    smart_contract_score: float
    economic_score: float
    oracle_score: float
    governance_score: float
    # Risk analysis
    critical_risks: list[str]
    high_risks: list[str]
    medium_risks: list[str]
    strengths: list[str]
    regulatory_flags: list[str]
    metrics_summary: dict
```

### Scoring Weights

| Dimension | Weight |
|-----------|--------|
| Smart Contract | 35% |
| Economic | 25% |
| Oracle | 20% |
| Governance | 20% |

### Pre-configured Protocols

`DEFI_PROTOCOL_DEFAULTS` contains configurations for:
- Aave V3 (Lending, Grade A)
- Uniswap V3 (DEX, Grade A)
- Lido (Staking, Grade A)
- GMX (Derivatives, Grade B)

### Functions

#### `score_defi_protocol()`
```python
def score_defi_protocol(
    protocol_id: str,
    category: DeFiCategory,
    smart_contract: SmartContractRisk,
    economic: EconomicRisk,
    oracle: OracleRisk,
    governance: GovernanceRisk,
) -> DeFiRiskScore
```

### Usage Example

```python
from backend.rule_service.app.services.defi_risk import (
    score_defi_protocol,
    DEFI_PROTOCOL_DEFAULTS,
    SmartContractRisk,
    EconomicRisk,
    OracleRisk,
    GovernanceRisk,
)

defaults = DEFI_PROTOCOL_DEFAULTS["aave_v3"]
score = score_defi_protocol(
    protocol_id="aave_v3",
    category=defaults["category"],
    smart_contract=SmartContractRisk(**defaults["smart_contract"]),
    economic=EconomicRisk(**defaults["economic"]),
    oracle=OracleRisk(**defaults["oracle"]),
    governance=GovernanceRisk(**defaults["governance"]),
)

print(f"Overall Grade: {score.overall_grade.value}")
print(f"Overall Score: {score.overall_score:.1f}")
print(f"Regulatory Flags: {score.regulatory_flags}")
```

---

## 4. Market Risk Analytics Module

**Location:** `rule_service/app/services/market_risk/volatility.py`

### Models

#### `CryptoVolatilityMetrics`
```python
class CryptoVolatilityMetrics(BaseModel):
    asset_id: str
    var_95_1d: float       # 1-day VaR at 95% confidence
    var_99_1d: float       # 1-day VaR at 99% confidence
    cvar_95_1d: float      # Conditional VaR (Expected Shortfall)
    cvar_99_1d: float
    volatility_30d: float  # 30-day annualized volatility
    volatility_90d: float
    max_drawdown_90d: float
    correlation_btc: float
    correlation_eth: float
    correlation_sp500: float
    beta_to_btc: float
```

#### `LiquidityMetrics`
```python
class LiquidityMetrics(BaseModel):
    asset_id: str
    bid_ask_spread_bps: float
    order_book_depth_1pct: float   # USD depth within 1% of mid
    order_book_depth_2pct: float
    avg_daily_volume_usd: float
    volume_volatility_ratio: float
    price_impact_100k_bps: float   # Slippage for $100k trade
    price_impact_1m_bps: float     # Slippage for $1M trade
```

#### `MarketRiskReport`
```python
class MarketRiskReport(BaseModel):
    asset_id: str
    timestamp: datetime
    volatility: CryptoVolatilityMetrics
    liquidity: LiquidityMetrics
    risk_rating: str  # LOW, MEDIUM, HIGH, VERY_HIGH
    risk_factors: list[str]
    position_recommendations: list[str]
```

### Functions

#### `calculate_volatility_metrics()`
```python
def calculate_volatility_metrics(
    asset_id: str,
    price_history: list[float],
    benchmark_prices: Optional[dict[str, list[float]]] = None,
) -> CryptoVolatilityMetrics
```

#### `calculate_liquidity_metrics()`
```python
def calculate_liquidity_metrics(
    asset_id: str,
    order_book: dict,
    volume_history: list[float],
) -> LiquidityMetrics
```

#### `generate_market_risk_report()`
```python
def generate_market_risk_report(
    asset_id: str,
    price_history: list[float],
    order_book: dict,
    volume_history: list[float],
    position_size_usd: float,
) -> MarketRiskReport
```

---

## 5. API Endpoints

### Risk Router

**Location:** `core/api/routes_risk.py`

#### POST `/risk/assess`
Position risk assessment with VaR/CVaR calculations.

**Request:**
```json
{
  "asset_id": "BTC",
  "position_size_usd": 1000000,
  "holding_period_days": 10,
  "confidence_level": 0.99
}
```

**Response:**
```json
{
  "asset_id": "BTC",
  "var_1d": 45000.0,
  "var_holding_period": 142302.5,
  "cvar_1d": 58500.0,
  "risk_rating": "MEDIUM",
  "risk_factors": ["High correlation to macro risk-off events"],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET `/risk/market-intelligence/{asset}`
Market data and risk indicators for an asset.

**Response:**
```json
{
  "asset_id": "ETH",
  "current_price": 2450.00,
  "volatility_30d": 0.65,
  "liquidity_score": 0.85,
  "correlation_btc": 0.82,
  "risk_tier": "MEDIUM"
}
```

### Navigate Router Enhancement

**Location:** `core/api/routes_navigate.py`

The `/navigate` endpoint now accepts additional fields:

**Request Enhancement:**
```json
{
  "issuer_jurisdiction": "US",
  "target_jurisdictions": ["EU", "UK"],
  "instrument_type": "stablecoin",
  "activity": "public_offer",
  "token_standard": "ERC-20",
  "underlying_chain": "ethereum",
  "is_defi_integrated": true,
  "defi_protocol": "aave_v3"
}
```

**Response Enhancement:**
```json
{
  "status": "actionable",
  "applicable_jurisdictions": [...],
  "token_compliance": {
    "standard": "erc_20",
    "classification": "payment_stablecoin",
    "requires_sec_registration": false,
    "genius_act_applicable": true,
    "howey_analysis": {...},
    "genius_analysis": {...}
  },
  "protocol_risk": {
    "protocol_id": "ethereum",
    "risk_tier": "tier_1",
    "overall_score": 80.2,
    "consensus_score": 85.0,
    "decentralization_score": 78.5
  },
  "defi_risk": {
    "protocol_id": "aave_v3",
    "overall_grade": "A",
    "overall_score": 88.7,
    "smart_contract_grade": "A",
    "regulatory_flags": []
  }
}
```

---

## 6. Jurisdiction Enhancements

### New Jurisdiction Codes

**Location:** `core/ontology/jurisdiction.py`

| Code | Description |
|------|-------------|
| `US_SEC` | US Securities and Exchange Commission |
| `US_CFTC` | US Commodity Futures Trading Commission |
| `HK` | Hong Kong |
| `JP` | Japan |

### Updated Default Regimes

**Location:** `rule_service/app/services/jurisdiction/resolver.py`

```python
DEFAULT_REGIMES = {
    "EU": "mica_2023",
    "UK": "fca_crypto_2024",
    "US": "securities_act_1933",
    "US_SEC": "securities_act_1933",
    "US_CFTC": "cftc_digital_assets_2024",
    "CH": "finsa_dlt_2021",
    "SG": "psa_2019",
    "HK": "sfc_vasp_2023",
    "JP": "psa_japan_2023",
}
```

---

## 7. Testing

### Unit Test Examples

```python
# Test protocol risk
def test_ethereum_tier_1():
    defaults = get_protocol_defaults("ethereum")
    result = assess_protocol_risk(protocol_id="ethereum", **defaults)
    assert result.risk_tier == RiskTier.TIER_1
    assert result.overall_score >= 75.0

# Test DeFi risk
def test_aave_grade_a():
    defaults = DEFI_PROTOCOL_DEFAULTS["aave_v3"]
    result = score_defi_protocol(
        protocol_id="aave_v3",
        category=defaults["category"],
        smart_contract=SmartContractRisk(**defaults["smart_contract"]),
        economic=EconomicRisk(**defaults["economic"]),
        oracle=OracleRisk(**defaults["oracle"]),
        governance=GovernanceRisk(**defaults["governance"]),
    )
    assert result.overall_grade == RiskGrade.A
    assert result.overall_score >= 85.0

# Test token compliance
def test_stablecoin_not_security():
    result = analyze_token_compliance(
        standard=TokenStandard.ERC_20,
        has_profit_expectation=False,
        is_decentralized=True,
        backed_by_fiat=True,
        is_stablecoin=True,
    )
    assert result.classification == TokenClassification.PAYMENT_STABLECOIN
    assert not result.requires_sec_registration
```

### Integration Test

```bash
cd backend
PYTHONPATH=.. python -c "
from backend.rule_service.app.services.protocol_risk import assess_protocol_risk, get_protocol_defaults
from backend.rule_service.app.services.defi_risk import score_defi_protocol, DEFI_PROTOCOL_DEFAULTS, SmartContractRisk, EconomicRisk, OracleRisk, GovernanceRisk
from backend.rule_service.app.services.token_standards import analyze_token_compliance, TokenStandard

# Verify all modules load and function
defaults = get_protocol_defaults('ethereum')
print(f'Protocol: {assess_protocol_risk(protocol_id=\"ethereum\", **defaults).risk_tier.value}')

d = DEFI_PROTOCOL_DEFAULTS['aave_v3']
print(f'DeFi: {score_defi_protocol(protocol_id=\"aave_v3\", category=d[\"category\"], smart_contract=SmartContractRisk(**d[\"smart_contract\"]), economic=EconomicRisk(**d[\"economic\"]), oracle=OracleRisk(**d[\"oracle\"]), governance=GovernanceRisk(**d[\"governance\"])).overall_grade.value}')

print(f'Token: {analyze_token_compliance(standard=TokenStandard.ERC_20, has_profit_expectation=False, is_decentralized=True, backed_by_fiat=True, is_stablecoin=True).classification.value}')
"
```

---

## 8. Architecture Decisions

### Risk Scoring Philosophy

1. **Quantitative Over Qualitative** - All risk dimensions produce numeric scores (0-100) that can be weighted and aggregated.

2. **Tiered Classification** - Protocol risk uses 4 tiers (institutional to high-risk); DeFi uses letter grades (A-F) for intuitive reporting.

3. **Regulatory Awareness** - Each module generates regulatory flags relevant to SEC, CFTC, and international frameworks.

4. **Configurable Defaults** - Pre-configured assessments for major protocols enable quick evaluation while allowing custom inputs.

### Integration Points

- **Navigate Endpoint** - Risk assessments are optionally included based on request parameters
- **Audit Trail** - All risk assessments are logged to the response audit trail
- **Error Handling** - Risk assessment failures are logged but don't block the primary navigation flow

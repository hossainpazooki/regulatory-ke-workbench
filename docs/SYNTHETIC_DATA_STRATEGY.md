# Synthetic Data Strategy

## Overview

This document describes the synthetic data generation strategy for expanding test coverage from ~30 scenarios to 500, and from ~24 rules to 75-100 across MiCA, FCA, GENIUS Act, and RWA frameworks.

## Regulatory Accuracy Assessment

| Framework | Status | Accuracy | Notes |
|-----------|--------|----------|-------|
| **MiCA (EU)** | Fully in force | **High** | Regulation (EU) 2023/1114, fully effective Jan 2025, 90+ CASPs authorized |
| **FCA Crypto (UK)** | Transition period | **High** | Draft Cryptoassets Regulations 2025, full regime effective late 2026 |
| **GENIUS Act (US)** | Enacted law | **High** | Signed July 2025, full compliance required by 2027 |
| **MAS PSA (SG)** | Enacted law | **High** | Payment Services Act, DPT licensing since 2020, stablecoin framework Aug 2023 |
| **FINMA DLT (CH)** | Enacted law | **High** | DLT Act 2021, Crypto Institution license proposal Oct 2025 |
| **RWA (EU)** | Hypothetical | **Low** | Explicitly marked "hypothetical rwa_eu_2025" |

**Implication:** Synthetic rule generation now covers all 5 active jurisdictions (EU, UK, US, SG, CH) with high-accuracy enacted frameworks, plus RWA as an illustrative hypothetical.

## Jurisdiction-Specific Requirements

### European Union (MiCA)

| Requirement | Threshold/Value | Article |
|-------------|-----------------|---------|
| Small offering exemption | €1,000,000 | Art. 4 |
| Significant ART threshold | €5,000,000 reserve value | Art. 43 |
| Reserve ratio | 100% (1:1 backing) | Art. 38 |
| CASP authorization | Required for all services | Art. 59-60 |
| Whitepaper publication | Mandatory for public offers | Art. 16-17 |
| Algorithmic stablecoins | Prohibited (no reserve = banned) | Art. 38 |

**Key MiCA Articles Covered:** Art. 3-4 (Definitions/Scope), Art. 16-17 (Whitepaper), Art. 36 (ART Authorization), Art. 38 (Reserve Assets), Art. 43-48 (Significant ARTs/EMTs), Art. 59-60 (CASP), Art. 86 (Market Abuse)

### United Kingdom (FCA Crypto)

| Requirement | Threshold/Value | Source |
|-------------|-----------------|--------|
| Financial promotions | FCA authorization required | COBS 4.12A |
| Risk warning | Prescribed format, prominent | PS22/10 |
| Cooling-off period | 24 hours for first-time investors | COBS 4.12A |
| MLR registration | AML supervision required | MLR 2017 |
| Stablecoin oversight | Bank of England (systemic) | Draft 2025 regs |

**UK Approach:** "Same risk, same regulatory outcome" - DeFi with identifiable controller treated as regulated.

### United States (GENIUS Act)

| Requirement | Threshold/Value | Source |
|-------------|-----------------|--------|
| Stablecoin reserve | 100% high-quality liquid assets | GENIUS Act §4 |
| Issuer licensing | Federal or state required | GENIUS Act §3 |
| Foreign stablecoin access | Must meet equivalent standards | GENIUS Act §7 |
| Audit requirements | Regular independent audits | GENIUS Act §5 |
| Full compliance deadline | 2027 | GENIUS Act §12 |

**US Approach:** Stablecoins under banking-like oversight; other digital assets regulated by function (securities vs commodities).

### Singapore (MAS PSA)

| Requirement | Threshold/Value | Source |
|-------------|-----------------|--------|
| DPT license base capital | S$250,000 (Major PI) | PSA 2019 |
| Cold storage ratio | 90% of client assets | MAS Notice |
| Retail lending/staking | Prohibited | MAS Guidelines |
| Stablecoin issuer capital | S$1,000,000 minimum | Stablecoin Framework |
| Stablecoin reserves | 100% (1:1) in cash/liquid assets | Stablecoin Framework |
| Redemption guarantee | At par within 5 business days | Stablecoin Framework |

**Singapore Approach:** Strict retail protection with no leverage/margin trading; only MAS-regulated stablecoins get official label.

### Switzerland (FINMA DLT)

| Requirement | Threshold/Value | Source |
|-------------|-----------------|--------|
| Token classification | Payment/Utility/Asset | FINMA Guidance 2018 |
| Travel Rule threshold | CHF 1,000 | FINMA Circular |
| DLT Trading Facility | FINMA license required | DLT Act 2021 |
| Crypto Institution license | Proposed for trading/custody | Draft Oct 2025 |
| Stablecoin issuer | Payment Institution license | Proposed framework |

**Swiss Approach:** Functional token classification; asset tokens = securities regulation; payment tokens = AML only.

## Database Storage Requirements

### Per-Record Sizes

| Record Type | Avg Size |
|-------------|----------|
| RuleRecord | ~3 KB |
| RuleVersionRecord | ~3 KB |
| RuleEventRecord | ~700 B |
| VerificationResultRecord | ~200 B |
| VerificationEvidenceRecord | ~450 B |
| EmbeddingRecord | ~1.7 KB |
| GraphNode | ~300 B |
| GraphEdge | ~200 B |
| Scenario (fixture) | ~300 B |

### Projected Storage (Target Volumes)

| Data Type | Count | Size |
|-----------|-------|------|
| Rules | 100 | 300 KB |
| Rule Versions (2/rule) | 200 | 600 KB |
| Rule Events (3/rule) | 300 | 210 KB |
| Verification Results | 200 | 40 KB |
| Verification Evidence | 400 | 180 KB |
| Embeddings (5 types x 100 rules) | 500 | 850 KB |
| Graph Nodes (5/rule) | 500 | 150 KB |
| Graph Edges (10/rule) | 1,000 | 200 KB |
| Scenarios | 500 | 150 KB |
| **Total** | - | **~2.7 MB** |

**Railway Compatibility:** Free tier (500 MB) = 0.5% utilization

## Package Structure

```
backend/synthetic_data/
├── __init__.py              # Package exports
├── base.py                  # BaseGenerator class
├── config.py                # Configuration and constants
├── scenario_generator.py    # ScenarioGenerator
├── rule_generator.py        # RuleGenerator
└── verification_generator.py # VerificationGenerator
```

## Generators

### ScenarioGenerator

Generates test scenarios by combining ontology dimensions:

```python
from backend.synthetic_data import ScenarioGenerator

generator = ScenarioGenerator(seed=42)
scenarios = generator.generate(count=500)

# Generate by category
happy_paths = generator.generate_category("happy_path", count=150)
edge_cases = generator.generate_category("edge_case", count=150)
```

**Scenario Categories:**

| Category | Count | Description |
|----------|-------|-------------|
| `happy_path` | 120 | Valid compliant scenarios |
| `edge_case` | 120 | Threshold boundaries |
| `negative` | 80 | Rule violations |
| `cross_border` | 60 | Multi-jurisdiction |
| `temporal` | 25 | Version-dependent |
| `stablecoin` | 30 | ART/EMT/algorithmic distinctions |
| `defi` | 20 | Decentralized protocol scenarios |
| `exemption` | 25 | Small offering, NFT, private placement |
| `aml_travel_rule` | 10 | KYC/originator data requirements |
| `passporting` | 10 | EU CASP passporting scenarios |

**Dimensions Covered:**

- `instrument_type`: art, emt, stablecoin, utility_token, security_token, nft, other_crypto
- `activity`: public_offer, admission_to_trading, custody, exchange, execution, placement, transfer, advice, portfolio_management
- `jurisdiction`: EU, UK, US, CH, SG
- `authorized`: True/False
- `is_significant`: True/False
- `is_credit_institution`: True/False
- `stablecoin_type`: art, emt, algorithmic, single_currency, multi_currency
- `defi_type`: centralized, decentralized_with_controller, fully_autonomous
- `investor_type`: retail, institutional, accredited, qualified
- `offering_type`: public, private_placement, qualified_investors_only, small_scale
- `license_type`: full_authorization, registration_only, sandbox, exempt

**Threshold Values Tested:**

```python
THRESHOLDS = {
    # MiCA (EU)
    "mica_small_offering_eur": [999_999, 1_000_000, 1_000_001],      # Small offering exemption
    "mica_significant_art_eur": [4_999_999, 5_000_000, 5_000_001],   # Significant ART threshold
    "mica_reserve_ratio": [0.99, 1.0, 1.01],                         # 100% reserve requirement

    # Singapore (MAS)
    "mas_base_capital_sgd": [249_999, 250_000, 250_001],             # DPT license capital
    "mas_stablecoin_capital_sgd": [999_999, 1_000_000, 1_000_001],   # Stablecoin issuer capital
    "mas_cold_storage_ratio": [0.89, 0.90, 0.91],                    # 90% cold storage requirement

    # Switzerland (FINMA)
    "finma_travel_rule_chf": [999, 1_000, 1_001],                    # Travel Rule threshold

    # UK (FCA)
    "fca_cooling_off_hours": [23, 24, 25],                           # 24h cooling-off period

    # US (GENIUS Act)
    "genius_reserve_ratio": [0.99, 1.0, 1.01],                       # 100% reserve mandatory

    # Legacy (backward compatibility)
    "reserve_value_eur": [4_999_999, 5_000_000, 5_000_001],
    "total_token_value_eur": [999_999, 1_000_000, 100_000_000],
    "reserve_ratio": [0.99, 1.0, 1.01],
}
```

### RuleGenerator

Generates YAML-compatible rule definitions:

```python
from backend.synthetic_data import RuleGenerator

generator = RuleGenerator(seed=42)
rules = generator.generate(count=50)

# Generate for specific framework
mica_rules = generator.generate_framework("mica_eu", count=15)
```

**Framework Distribution:**

| Framework | Count | Accuracy | Key Articles/Rules |
|-----------|-------|----------|-------------------|
| MiCA (EU) | 20-25 | High | Art. 3-4, 16-17, 36, 38, 43-48, 59-60, 86 |
| FCA (UK) | 15-18 | High | COBS 4.12A, PS22/10, FSMA 2023, draft 2025 regs |
| GENIUS Act (US) | 15-18 | High | Reserve requirements, issuer licensing, foreign access |
| MAS PSA (SG) | 12-15 | High | DPT licensing, custody, stablecoin framework |
| FINMA DLT (CH) | 10-12 | High | Token classification, DLT facilities, Travel Rule |
| RWA Tokenization | 8-10 | Low (hypothetical) | Hypothetical framework |

**Complexity Levels:**

| Level | Percentage | Description |
|-------|------------|-------------|
| Simple | 30% | Single condition -> single outcome |
| Medium | 50% | 2-3 nested conditions |
| Complex | 20% | Multi-branch decision trees |

### VerificationGenerator

Generates verification evidence records:

```python
from backend.synthetic_data import VerificationGenerator

generator = VerificationGenerator(seed=42)
evidence = generator.generate(count=200)

# Generate for specific tier
tier0 = generator.generate_tier(tier=0, count=80)
```

**Verification Tiers:**

| Tier | Category | Distribution |
|------|----------|--------------|
| 0 | Schema validation | 40% |
| 1 | Semantic consistency | 25% |
| 2 | Cross-rule checks | 15% |
| 3 | Temporal consistency | 10% |
| 4 | External alignment | 10% |

**Confidence Score Ranges:**

| Outcome | Score Range |
|---------|-------------|
| Passing | 0.85-0.99 |
| Marginal | 0.70-0.84 |
| Failing | 0.40-0.69 |

## Stablecoin-Specific Scenarios

The `stablecoin` category (30 scenarios) tests regulatory distinctions between stablecoin types:

### Asset-Referenced Tokens (ARTs)

| Scenario | Expected Outcome | Jurisdiction |
|----------|------------------|--------------|
| Single-asset ART with 100% reserves | compliant | EU (MiCA) |
| Basket-referenced ART (multi-asset) | requires_authorization | EU (MiCA) |
| Significant ART (>€5M reserve value) | enhanced_requirements | EU (MiCA) |
| ART without authorized custodian | non_compliant | EU (MiCA) |
| Volume cap exceeded | restricted | EU (MiCA) |

### E-Money Tokens (EMTs)

| Scenario | Expected Outcome | Jurisdiction |
|----------|------------------|--------------|
| EMT issued by credit institution | exempt (from separate license) | EU (MiCA) |
| EMT single-currency (EUR-pegged) | compliant | EU (MiCA) |
| Significant EMT (>€5M) | enhanced_requirements | EU (MiCA) |
| EMT without e-money license | non_compliant | EU (MiCA) |

### Algorithmic Stablecoins

| Scenario | Expected Outcome | Jurisdiction |
|----------|------------------|--------------|
| Algorithmic (no reserve backing) | prohibited | EU (MiCA) |
| Partial reserve (<100%) | non_compliant | US (GENIUS Act) |
| Foreign stablecoin without equivalent standards | restricted | US (GENIUS Act) |

## Cross-Border Compliance Scenarios

The `cross_border` (60) and `passporting` (10) categories test multi-jurisdiction compliance:

### EU Passporting Scenarios

| Scenario | Expected Outcome |
|----------|------------------|
| CASP authorized in Germany → offers in France | compliant (passporting) |
| CASP authorized in Germany → offers in all EU | compliant (passporting) |
| Third-country firm → EU clients (no establishment) | non_compliant |
| Reverse solicitation claim (EU client initiated) | requires_review |

### Equivalence Scenarios

| Scenario | Expected Outcome |
|----------|------------------|
| UK firm → Swiss clients (mutual recognition) | compliant |
| US stablecoin meeting GENIUS standards → US market | compliant |
| Foreign stablecoin without equivalent oversight → US | restricted |
| Singapore firm → offshore clients (DTSP license) | requires_authorization |

### Conflict Scenarios

| Conflict Type | Jurisdictions | Resolution |
|---------------|---------------|------------|
| Risk warning format | EU vs US | Apply stricter (EU) |
| Cooling-off period | UK (24h) vs others (none) | Apply UK for UK-targeted |
| Whitepaper requirements | EU (detailed) vs US (varies) | Prepare for strictest |
| Reserve disclosure | MiCA vs GENIUS Act | Meet both standards |

## DeFi and NFT Testing

The `defi` (20) and `exemption` (25) categories test decentralized protocol and exemption scenarios:

### DeFi Classification Scenarios

| Scenario | Classification | Regulatory Outcome |
|----------|----------------|-------------------|
| DEX with identifiable development company | decentralized_with_controller | regulated (UK approach) |
| Fully autonomous protocol (no admin keys) | fully_autonomous | unregulated |
| DAO with token voting governance | decentralized_with_controller | case-by-case |
| DeFi lending with UK-based interface | decentralized_with_controller | FCA oversight |

### NFT Exception Scenarios

| Scenario | Expected Outcome | Jurisdiction |
|----------|------------------|--------------|
| Unique art NFT (one-of-a-kind) | exempt | EU (MiCA carve-out) |
| NFT series (1000 identical tokens) | subject_to_regulation | EU (fungible = regulated) |
| NFT fractionalized into shares | securities_law | US (Howey test) |
| Gaming NFT (in-game utility only) | exempt | Most jurisdictions |
| NFT marketed as investment | securities_law | US, UK, EU |

### Small Offering Exemptions

| Scenario | Threshold | Jurisdiction |
|----------|-----------|--------------|
| Token offering <€1M (12 months) | Exempt from whitepaper | EU (MiCA Art. 4) |
| Private placement to accredited investors | Reg D exemption | US |
| Qualified investor only offering | Reduced requirements | UK |

## AML/Travel Rule Scenarios

The `aml_travel_rule` category (10 scenarios) tests anti-money laundering compliance:

### Travel Rule Compliance

| Scenario | Threshold | Jurisdiction | Outcome |
|----------|-----------|--------------|---------|
| Transfer >CHF 1,000 without originator data | CHF 1,000 | CH (FINMA) | non_compliant |
| Transfer with full originator/beneficiary info | Any amount | All | compliant |
| Unhosted wallet transfer (no beneficiary VASP) | Varies | EU, US | enhanced_due_diligence |
| Transfer involving high-risk jurisdiction | N/A | All | blocked or enhanced_review |
| Mixer/tumbler interaction detected | N/A | All | suspicious_activity_report |

### KYC Requirements

| Scenario | Jurisdiction | Outcome |
|----------|--------------|---------|
| Retail customer without KYC | SG (MAS) | non_compliant |
| Institutional client with simplified DD | EU, UK | compliant |
| PEP (Politically Exposed Person) detected | All | enhanced_due_diligence |
| Sanctions list match | All | blocked |

## Pytest Integration

### Fixtures Available

Session-scoped fixtures (generated once):

```python
@pytest.fixture(scope="session")
def synthetic_scenarios() -> list[dict]:
    """500 scenarios across all categories."""

@pytest.fixture(scope="session")
def synthetic_rules() -> list[dict]:
    """80-100 rules across all 6 frameworks."""

@pytest.fixture(scope="session")
def synthetic_verification() -> list[dict]:
    """200 evidence records across all tiers."""
```

Category-filtered fixtures:

```python
@pytest.fixture
def happy_path_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only happy path scenarios."""

@pytest.fixture
def edge_case_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only edge case scenarios."""

@pytest.fixture
def negative_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only negative scenarios."""

@pytest.fixture
def stablecoin_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only stablecoin (ART/EMT/algorithmic) scenarios."""

@pytest.fixture
def defi_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only DeFi classification scenarios."""

@pytest.fixture
def exemption_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only exemption (small offering, NFT, private placement) scenarios."""

@pytest.fixture
def aml_travel_rule_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only AML/Travel Rule compliance scenarios."""

@pytest.fixture
def passporting_scenarios(synthetic_scenarios) -> list[dict]:
    """Filter to only EU CASP passporting scenarios."""
```

Parametrized fixture for testing across all 10 categories:

```python
@pytest.fixture(params=[
    "happy_path", "edge_case", "negative", "cross_border", "temporal",
    "stablecoin", "defi", "exemption", "aml_travel_rule", "passporting"
])
def scenario_category(request, synthetic_scenarios) -> list[dict]:
    """Parametrized fixture for all categories."""
```

### Example Test Using Fixtures

```python
def test_decision_engine_handles_edge_cases(decision_engine, edge_case_scenarios):
    """Test decision engine with threshold boundary scenarios."""
    for scenario in edge_case_scenarios[:10]:
        result = decision_engine.evaluate(scenario)
        assert result is not None
        # Verify boundary handling
        if scenario.get("reserve_ratio", 1.0) < 1.0:
            assert result.decision == "non_compliant"
```

## CLI Usage

### Generate Scenarios

```bash
# Generate 100 scenarios
python -m backend.synthetic_data.scenario_generator --count 100 --seed 42

# Generate specific category
python -m backend.synthetic_data.scenario_generator --category edge_case --count 50

# Output to file
python -m backend.synthetic_data.scenario_generator --count 500 --output scenarios.json
```

### Generate Rules

```bash
# Generate 50 rules
python -m backend.synthetic_data.rule_generator --count 50 --seed 42

# Generate for specific framework
python -m backend.synthetic_data.rule_generator --framework mica_eu --count 15

# Output as YAML
python -m backend.synthetic_data.rule_generator --count 50 --format yaml --output rules.yaml
```

### Generate Verification Evidence

```bash
# Generate 100 evidence records
python -m backend.synthetic_data.verification_generator --count 100 --seed 42

# Generate for specific tier
python -m backend.synthetic_data.verification_generator --tier 0 --count 50

# Output to file
python -m backend.synthetic_data.verification_generator --count 200 --output evidence.json
```

## Running Tests

```bash
# Run all synthetic data tests
pytest tests/test_synthetic_coverage.py -v

# Run with coverage
pytest tests/test_synthetic_coverage.py -v --cov=backend/synthetic_data

# Run specific test class
pytest tests/test_synthetic_coverage.py::TestScenarioGenerator -v

# Validate generated data
python -m backend.synthetic_data.scenario_generator --count 100 --validate
python -m backend.synthetic_data.rule_generator --count 50 --validate
```

## Data Volume Targets

| Data Type | Current | Target | Growth |
|-----------|---------|--------|--------|
| Rules | 24 | 80-100 | 3-4x |
| Frameworks | 4 | 6 | 1.5x |
| Scenarios | 30 | 500 | 16x |
| Scenario Categories | 5 | 10 | 2x |
| Jurisdictions (active) | 3 | 5 | 1.7x |
| Verification Evidence | ~10 | 200 | 20x |
| Ontology Coverage | ~60% | 100% | - |

## Key Design Decisions

1. **Deterministic Generation**: All generators use seeded random for reproducibility
2. **Session-Scoped Fixtures**: Generated once per test session for performance
3. **Accuracy Labels**: Rules clearly labeled as high/medium/low accuracy based on enacted vs proposed law
4. **Category Filtering**: Easy access to specific scenario types (10 categories)
5. **Ontology Alignment**: Generated data matches `backend/core/ontology/types.py`
6. **YAML Compatibility**: Generated rules match existing rule structure
7. **Global Coverage**: All 5 jurisdictions (EU, UK, US, SG, CH) with enacted frameworks
8. **2026 Regulatory Alignment**: Thresholds and requirements match current enacted law (MiCA, GENIUS Act, MAS PSA, FINMA DLT)

## Future Enhancements

### Completed (2026 Update)
- [x] Add cross-jurisdiction conflict scenarios (passporting, equivalence)
- [x] Add stablecoin-specific scenarios (ART/EMT/algorithmic)
- [x] Add DeFi and NFT exception scenarios
- [x] Add AML/Travel Rule compliance scenarios
- [x] Add Singapore (MAS PSA) and Switzerland (FINMA DLT) frameworks

### Pending
- [ ] Add embedding generation for synthetic rules
- [ ] Generate graph structures for rule relationships
- [ ] Generate temporal version chains
- [ ] Add natural language variation to rule descriptions
- [ ] Implement code support for new scenario categories (stablecoin, defi, exemption, aml_travel_rule, passporting)
- [ ] Add GENIUS Act foreign stablecoin access testing
- [ ] Add UK comprehensive regime scenarios (late 2026)

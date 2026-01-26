"""Tests for DeFi risk scoring domain."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.main import app
from backend.defi_risk import (
    DeFiCategory,
    GovernanceType,
    OracleProvider,
    RiskGrade,
    SmartContractRisk,
    EconomicRisk,
    OracleRisk,
    GovernanceRisk,
    DeFiRiskScore,
    DeFiScoreRequest,
    score_defi_protocol,
    get_protocol_defaults,
    list_protocol_defaults,
    DEFI_PROTOCOL_DEFAULTS,
    REPUTABLE_AUDITORS,
)


@pytest.fixture
def client():
    """Test client for the FastAPI app."""
    return TestClient(app)


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestDeFiScoreRequestValidation:
    """Tests for DeFiScoreRequest input validation."""

    def test_valid_protocol_id(self):
        """Valid protocol IDs should pass validation."""
        valid_ids = ["aave_v3", "uniswap-v3", "GMX", "compound_v2", "test123"]
        for protocol_id in valid_ids:
            request = DeFiScoreRequest(
                protocol_id=protocol_id,
                category=DeFiCategory.LENDING,
            )
            assert request.protocol_id == protocol_id

    def test_invalid_protocol_id_empty(self):
        """Empty protocol ID should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            DeFiScoreRequest(protocol_id="", category=DeFiCategory.LENDING)
        assert "protocol_id" in str(exc_info.value)

    def test_invalid_protocol_id_special_chars(self):
        """Protocol ID with special chars should fail validation."""
        with pytest.raises(ValidationError):
            DeFiScoreRequest(protocol_id="aave@v3!", category=DeFiCategory.LENDING)

    def test_invalid_protocol_id_too_long(self):
        """Protocol ID over 100 chars should fail validation."""
        with pytest.raises(ValidationError):
            DeFiScoreRequest(protocol_id="a" * 101, category=DeFiCategory.LENDING)

    def test_smart_contract_risk_defaults(self):
        """SmartContractRisk should have sensible defaults."""
        risk = SmartContractRisk()
        assert risk.audit_count == 0
        assert risk.admin_can_drain is False
        assert risk.is_upgradeable is True

    def test_smart_contract_risk_validation(self):
        """SmartContractRisk fields should validate ranges."""
        with pytest.raises(ValidationError):
            SmartContractRisk(audit_count=-1)
        with pytest.raises(ValidationError):
            SmartContractRisk(tvl_usd=-100)

    def test_economic_risk_percentage_validation(self):
        """Economic risk percentages should be 0-100."""
        with pytest.raises(ValidationError):
            EconomicRisk(token_concentration_top10_pct=150)
        with pytest.raises(ValidationError):
            EconomicRisk(team_token_pct=-10)

    def test_oracle_risk_defaults(self):
        """OracleRisk should default to Chainlink."""
        risk = OracleRisk()
        assert risk.primary_oracle == OracleProvider.CHAINLINK

    def test_governance_risk_multisig_threshold(self):
        """GovernanceRisk should accept multisig threshold."""
        risk = GovernanceRisk(
            governance_type=GovernanceType.MULTISIG,
            multisig_threshold="3/5",
        )
        assert risk.multisig_threshold == "3/5"


# =============================================================================
# Service Layer Tests
# =============================================================================


class TestDeFiRiskScoring:
    """Tests for DeFi risk scoring service."""

    def test_score_high_risk_protocol(self):
        """Protocol with no audits should get poor grade."""
        score = score_defi_protocol(
            protocol_id="high_risk_protocol",
            category=DeFiCategory.LENDING,
            smart_contract=SmartContractRisk(
                audit_count=0,
                admin_can_drain=True,
                exploit_history_count=2,
                total_exploit_loss_usd=50_000_000,
            ),
            economic=EconomicRisk(
                token_concentration_top10_pct=90,
                treasury_runway_months=6,
            ),
            oracle=OracleRisk(
                primary_oracle=OracleProvider.CUSTOM,
                has_fallback_oracle=False,
                oracle_manipulation_resistant=False,
            ),
            governance=GovernanceRisk(
                governance_type=GovernanceType.CENTRALIZED,
                has_timelock=False,
            ),
        )

        assert score.overall_grade in [RiskGrade.D, RiskGrade.F]
        assert len(score.critical_risks) > 0
        assert "admin can drain" in " ".join(score.critical_risks).lower()

    def test_score_low_risk_protocol(self):
        """Well-audited protocol with good governance should get high grade."""
        score = score_defi_protocol(
            protocol_id="low_risk_protocol",
            category=DeFiCategory.DEX,
            smart_contract=SmartContractRisk(
                audit_count=5,
                auditors=["trail of bits", "openzeppelin", "certik"],
                formal_verification=True,
                is_upgradeable=False,
                admin_can_drain=False,
                tvl_usd=5_000_000_000,
                contract_age_days=500,
                bug_bounty_max_usd=2_000_000,
            ),
            economic=EconomicRisk(
                token_concentration_top10_pct=30,
                treasury_runway_months=48,
                treasury_diversified=True,
                has_protocol_revenue=True,
                revenue_30d_usd=5_000_000,
            ),
            oracle=OracleRisk(
                primary_oracle=OracleProvider.NONE,  # DEX doesn't need oracle
            ),
            governance=GovernanceRisk(
                governance_type=GovernanceType.TOKEN_VOTING,
                has_timelock=True,
                timelock_hours=72,
                governance_participation_pct=20,
            ),
        )

        assert score.overall_grade in [RiskGrade.A, RiskGrade.B]
        assert len(score.strengths) > 0
        assert score.overall_score >= 70

    def test_regulatory_flags_for_lending(self):
        """Lending protocols should get state licensing flag."""
        score = score_defi_protocol(
            protocol_id="test_lending",
            category=DeFiCategory.LENDING,
            smart_contract=SmartContractRisk(),
            economic=EconomicRisk(),
            oracle=OracleRisk(),
            governance=GovernanceRisk(),
        )

        assert any("licensing" in flag.lower() for flag in score.regulatory_flags)

    def test_regulatory_flags_for_derivatives(self):
        """Derivatives protocols should get CFTC flag."""
        score = score_defi_protocol(
            protocol_id="test_derivatives",
            category=DeFiCategory.DERIVATIVES,
            smart_contract=SmartContractRisk(),
            economic=EconomicRisk(),
            oracle=OracleRisk(),
            governance=GovernanceRisk(),
        )

        assert any("cftc" in flag.lower() for flag in score.regulatory_flags)

    def test_regulatory_flags_for_stablecoin(self):
        """Stablecoin protocols should get GENIUS Act flag."""
        score = score_defi_protocol(
            protocol_id="test_stablecoin",
            category=DeFiCategory.STABLECOIN,
            smart_contract=SmartContractRisk(),
            economic=EconomicRisk(),
            oracle=OracleRisk(),
            governance=GovernanceRisk(),
        )

        assert any("genius" in flag.lower() for flag in score.regulatory_flags)

    def test_critical_risk_caps_grade(self):
        """Critical risks should cap the overall grade."""
        score = score_defi_protocol(
            protocol_id="critical_risk",
            category=DeFiCategory.LENDING,
            smart_contract=SmartContractRisk(
                audit_count=5,
                admin_can_drain=True,  # Critical risk
            ),
            economic=EconomicRisk(),
            oracle=OracleRisk(),
            governance=GovernanceRisk(),
        )

        # Critical risk should cap score at 35, resulting in F or D grade
        assert score.overall_grade in [RiskGrade.D, RiskGrade.F]
        assert score.overall_score <= 35

    def test_score_includes_metrics_summary(self):
        """Score should include metrics summary."""
        score = score_defi_protocol(
            protocol_id="test",
            category=DeFiCategory.DEX,
            smart_contract=SmartContractRisk(tvl_usd=1_000_000),
            economic=EconomicRisk(),
            oracle=OracleRisk(),
            governance=GovernanceRisk(governance_type=GovernanceType.MULTISIG),
        )

        assert "category" in score.metrics_summary
        assert "tvl_usd" in score.metrics_summary
        assert score.metrics_summary["category"] == "dex"


class TestProtocolDefaults:
    """Tests for protocol default configurations."""

    def test_list_protocol_defaults(self):
        """Should list available protocol defaults."""
        protocols = list_protocol_defaults()
        assert isinstance(protocols, list)
        assert "aave_v3" in protocols
        assert "uniswap_v3" in protocols
        assert "lido" in protocols
        assert "gmx" in protocols

    def test_get_aave_defaults(self):
        """Should get Aave v3 default configuration."""
        config = get_protocol_defaults("aave_v3")
        assert config is not None
        assert config["category"] == DeFiCategory.LENDING
        assert "smart_contract" in config
        assert config["smart_contract"]["audit_count"] >= 3

    def test_get_uniswap_defaults(self):
        """Should get Uniswap v3 default configuration."""
        config = get_protocol_defaults("uniswap_v3")
        assert config is not None
        assert config["category"] == DeFiCategory.DEX
        assert config["smart_contract"]["is_upgradeable"] is False

    def test_get_nonexistent_protocol(self):
        """Should return None for unknown protocol."""
        config = get_protocol_defaults("nonexistent_protocol")
        assert config is None

    def test_case_insensitive_lookup(self):
        """Protocol lookup should be case insensitive."""
        config1 = get_protocol_defaults("AAVE_V3")
        config2 = get_protocol_defaults("aave_v3")
        assert config1 == config2


class TestReputableAuditors:
    """Tests for auditor reputation scoring."""

    def test_top_tier_auditors(self):
        """Top tier auditors should have weight 1.0."""
        top_tier = ["trail of bits", "openzeppelin", "consensys diligence", "spearbit"]
        for auditor in top_tier:
            assert REPUTABLE_AUDITORS.get(auditor) == 1.0

    def test_competition_auditors(self):
        """Competition auditors should have weight 0.9."""
        competitions = ["code4rena", "sherlock"]
        for auditor in competitions:
            assert REPUTABLE_AUDITORS.get(auditor) == 0.9


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestDeFiRiskEndpoints:
    """Tests for DeFi risk API endpoints."""

    def test_score_protocol_endpoint(self, client):
        """POST /defi-risk/score should return risk score."""
        response = client.post(
            "/defi-risk/score",
            json={
                "protocol_id": "test_protocol",
                "category": "lending",
                "smart_contract": {"audit_count": 3},
                "economic": {"token_concentration_top10_pct": 40},
                "oracle": {"primary_oracle": "chainlink"},
                "governance": {"governance_type": "token_voting"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["protocol_id"] == "test_protocol"
        assert "overall_grade" in data
        assert "overall_score" in data
        assert data["overall_grade"] in ["A", "B", "C", "D", "F"]

    def test_score_protocol_validation_error(self, client):
        """POST /defi-risk/score should reject invalid input."""
        response = client.post(
            "/defi-risk/score",
            json={
                "protocol_id": "",  # Invalid: empty
                "category": "lending",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_list_protocols_endpoint(self, client):
        """GET /defi-risk/protocols should list defaults."""
        response = client.get("/defi-risk/protocols")

        assert response.status_code == 200
        data = response.json()
        assert "protocols" in data
        assert "aave_v3" in data["protocols"]

    def test_get_protocol_config_endpoint(self, client):
        """GET /defi-risk/protocols/{id} should return config."""
        response = client.get("/defi-risk/protocols/aave_v3")

        assert response.status_code == 200
        data = response.json()
        assert data["protocol_id"] == "aave_v3"
        assert "smart_contract" in data

    def test_get_protocol_config_not_found(self, client):
        """GET /defi-risk/protocols/{id} should 404 for unknown."""
        response = client.get("/defi-risk/protocols/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_score_known_protocol_endpoint(self, client):
        """POST /defi-risk/protocols/{id}/score should score known protocol."""
        response = client.post("/defi-risk/protocols/aave_v3/score")

        assert response.status_code == 200
        data = response.json()
        assert data["protocol_id"] == "aave_v3"
        assert data["category"] == "lending"
        # Aave should get a good grade with its defaults
        assert data["overall_grade"] in ["A", "B"]

    def test_score_known_protocol_not_found(self, client):
        """POST /defi-risk/protocols/{id}/score should 404 for unknown."""
        response = client.post("/defi-risk/protocols/nonexistent/score")

        assert response.status_code == 404

    def test_list_categories_endpoint(self, client):
        """GET /defi-risk/categories should list categories."""
        response = client.get("/defi-risk/categories")

        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "lending" in data["categories"]
        assert "dex" in data["categories"]
        assert "staking" in data["categories"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDeFiRiskEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_maximum_scores(self):
        """Test protocol with maximum positive inputs."""
        score = score_defi_protocol(
            protocol_id="max_score",
            category=DeFiCategory.DEX,
            smart_contract=SmartContractRisk(
                audit_count=10,
                auditors=["trail of bits", "openzeppelin", "spearbit"],
                formal_verification=True,
                is_upgradeable=False,
                admin_can_drain=False,
                admin_can_pause=False,
                tvl_usd=10_000_000_000,
                contract_age_days=1000,
                exploit_history_count=0,
                bug_bounty_max_usd=5_000_000,
            ),
            economic=EconomicRisk(
                token_concentration_top10_pct=20,
                treasury_runway_months=60,
                treasury_diversified=True,
                has_protocol_revenue=True,
                revenue_30d_usd=10_000_000,
            ),
            oracle=OracleRisk(
                primary_oracle=OracleProvider.NONE,
            ),
            governance=GovernanceRisk(
                governance_type=GovernanceType.IMMUTABLE,
            ),
        )

        assert score.overall_grade == RiskGrade.A
        assert score.overall_score >= 85

    def test_minimum_scores(self):
        """Test protocol with minimum/negative inputs."""
        score = score_defi_protocol(
            protocol_id="min_score",
            category=DeFiCategory.BRIDGE,
            smart_contract=SmartContractRisk(
                audit_count=0,
                admin_can_drain=True,
                exploit_history_count=5,
                total_exploit_loss_usd=100_000_000,
                contract_age_days=30,
            ),
            economic=EconomicRisk(
                token_concentration_top10_pct=95,
                treasury_runway_months=3,
            ),
            oracle=OracleRisk(
                primary_oracle=OracleProvider.CUSTOM,
                oracle_manipulation_resistant=False,
                oracle_failure_count_12m=5,
            ),
            governance=GovernanceRisk(
                governance_type=GovernanceType.CENTRALIZED,
                has_timelock=False,
                emergency_actions_12m=10,
            ),
        )

        assert score.overall_grade == RiskGrade.F
        assert score.overall_score < 40

    def test_all_categories(self):
        """Score should work for all DeFi categories."""
        for category in DeFiCategory:
            score = score_defi_protocol(
                protocol_id=f"test_{category.value}",
                category=category,
                smart_contract=SmartContractRisk(),
                economic=EconomicRisk(),
                oracle=OracleRisk(),
                governance=GovernanceRisk(),
            )
            assert score.category == category
            assert score.overall_grade in [RiskGrade.A, RiskGrade.B, RiskGrade.C, RiskGrade.D, RiskGrade.F]

    def test_all_governance_types(self):
        """Score should work for all governance types."""
        for gov_type in GovernanceType:
            score = score_defi_protocol(
                protocol_id="test",
                category=DeFiCategory.LENDING,
                smart_contract=SmartContractRisk(),
                economic=EconomicRisk(),
                oracle=OracleRisk(),
                governance=GovernanceRisk(governance_type=gov_type),
            )
            assert score.governance_grade in RiskGrade

    def test_all_oracle_providers(self):
        """Score should work for all oracle providers."""
        for oracle in OracleProvider:
            score = score_defi_protocol(
                protocol_id="test",
                category=DeFiCategory.LENDING,
                smart_contract=SmartContractRisk(),
                economic=EconomicRisk(),
                oracle=OracleRisk(primary_oracle=oracle),
                governance=GovernanceRisk(),
            )
            assert score.oracle_grade in RiskGrade

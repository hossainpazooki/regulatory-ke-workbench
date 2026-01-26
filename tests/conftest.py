"""Pytest fixtures for test suite."""

import pytest
from pathlib import Path
from typing import Any

from backend.core.ontology import Scenario
from backend.rules import RuleLoader, DecisionEngine
from backend.rag import Retriever, BM25Index


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture
def rules_dir() -> Path:
    """Path to the rules directory."""
    return Path(__file__).parent.parent / "backend" / "rules" / "data"


@pytest.fixture
def rule_loader(rules_dir: Path) -> RuleLoader:
    """Rule loader with rules loaded from the rules directory."""
    loader = RuleLoader(rules_dir)
    loader.load_directory()
    return loader


@pytest.fixture
def decision_engine(rule_loader: RuleLoader) -> DecisionEngine:
    """Decision engine with rules loaded."""
    return DecisionEngine(rule_loader)


@pytest.fixture
def sample_scenario() -> Scenario:
    """Sample scenario for testing."""
    return Scenario(
        instrument_type="art",
        activity="public_offer",
        jurisdiction="EU",
        authorized=False,
        is_credit_institution=False,
    )


@pytest.fixture
def bm25_index() -> BM25Index:
    """BM25 index with sample documents."""
    index = BM25Index()
    index.add_documents([
        {
            "id": "doc1",
            "text": "Article 36 requires authorization for public offers of asset-referenced tokens.",
            "metadata": {"source": "MiCA"},
        },
        {
            "id": "doc2",
            "text": "E-money tokens can only be issued by authorized credit institutions.",
            "metadata": {"source": "MiCA"},
        },
        {
            "id": "doc3",
            "text": "Reserve assets must be held by authorized custodians.",
            "metadata": {"source": "MiCA"},
        },
    ])
    return index


@pytest.fixture
def retriever() -> Retriever:
    """Retriever with sample documents (BM25 only)."""
    retriever = Retriever(use_vectors=False)
    retriever.add_documents([
        {
            "id": "mica_art36",
            "text": """Article 36 — Authorisation
            No person shall make a public offer in the Union of an asset-referenced token,
            or seek admission of such a crypto-asset to trading on a trading platform,
            unless that person is a legal person that has been authorised in accordance
            with Article 21 or is a credit institution.""",
            "metadata": {"article": "36"},
        },
        {
            "id": "mica_art38",
            "text": """Article 38 — Reserve of assets
            Issuers of asset-referenced tokens shall constitute and maintain a reserve
            of assets. The reserve shall be composed and managed in such a way that
            the risks associated with the assets referenced are covered.""",
            "metadata": {"article": "38"},
        },
    ])
    return retriever


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def synthetic_scenarios() -> list[dict[str, Any]]:
    """Generate synthetic test scenarios.

    Returns 500 scenarios distributed across categories:
    - Happy path (150): Valid compliant scenarios
    - Edge cases (150): Threshold boundaries
    - Negative cases (100): Rule violations
    - Cross-border (75): Multi-jurisdiction
    - Temporal (25): Version-dependent
    """
    from backend.synthetic_data import ScenarioGenerator
    return ScenarioGenerator(seed=42).generate(count=500)


@pytest.fixture(scope="session")
def synthetic_rules() -> list[dict[str, Any]]:
    """Generate synthetic rules for testing.

    Returns 50 rules distributed across frameworks:
    - MiCA (EU): High accuracy
    - FCA (UK): High accuracy
    - GENIUS Act (US): Medium accuracy (illustrative)
    - RWA Tokenization: Low accuracy (hypothetical)
    """
    from backend.synthetic_data import RuleGenerator
    return RuleGenerator(seed=42).generate(count=50)


@pytest.fixture(scope="session")
def synthetic_verification() -> list[dict[str, Any]]:
    """Generate synthetic verification evidence.

    Returns 200 evidence records distributed across tiers:
    - Tier 0 (40%): Schema validation
    - Tier 1 (25%): Semantic consistency
    - Tier 2 (15%): Cross-rule checks
    - Tier 3 (10%): Temporal consistency
    - Tier 4 (10%): External alignment
    """
    from backend.synthetic_data import VerificationGenerator
    return VerificationGenerator(seed=42).generate(count=200)


# =============================================================================
# Scenario Category Fixtures
# =============================================================================


@pytest.fixture
def happy_path_scenarios(synthetic_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to only happy path scenarios."""
    return [s for s in synthetic_scenarios if s.get("category") == "happy_path"]


@pytest.fixture
def edge_case_scenarios(synthetic_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to only edge case scenarios."""
    return [s for s in synthetic_scenarios if s.get("category") == "edge_case"]


@pytest.fixture
def negative_scenarios(synthetic_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to only negative (violation) scenarios."""
    return [s for s in synthetic_scenarios if s.get("category") == "negative"]


@pytest.fixture
def cross_border_scenarios(synthetic_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to only cross-border scenarios."""
    return [s for s in synthetic_scenarios if s.get("category") == "cross_border"]


@pytest.fixture
def temporal_scenarios(synthetic_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to only temporal scenarios."""
    return [s for s in synthetic_scenarios if s.get("category") == "temporal"]


# =============================================================================
# Parametrized Scenario Fixture
# =============================================================================


@pytest.fixture(params=["happy_path", "edge_case", "negative", "cross_border", "temporal"])
def scenario_category(request, synthetic_scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parametrized fixture for testing across all scenario categories."""
    return [s for s in synthetic_scenarios if s.get("category") == request.param]


# =============================================================================
# Framework-Specific Rule Fixtures
# =============================================================================


@pytest.fixture
def mica_rules(synthetic_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to MiCA (EU) rules only."""
    return [r for r in synthetic_rules if r.get("framework") == "MiCA"]


@pytest.fixture
def fca_rules(synthetic_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to FCA (UK) rules only."""
    return [r for r in synthetic_rules if r.get("framework") == "FCA Crypto"]


@pytest.fixture
def genius_rules(synthetic_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to GENIUS Act (US) rules only."""
    return [r for r in synthetic_rules if r.get("framework") == "GENIUS Act"]


@pytest.fixture
def rwa_rules(synthetic_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to RWA Tokenization rules only."""
    return [r for r in synthetic_rules if r.get("framework") == "RWA Tokenization"]


# =============================================================================
# Verification Tier Fixtures
# =============================================================================


@pytest.fixture
def tier0_verification(synthetic_verification: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to Tier 0 (schema validation) evidence."""
    return [v for v in synthetic_verification if v.get("tier") == 0]


@pytest.fixture
def tier1_verification(synthetic_verification: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to Tier 1 (semantic consistency) evidence."""
    return [v for v in synthetic_verification if v.get("tier") == 1]


@pytest.fixture
def passing_verification(synthetic_verification: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to passing verification evidence."""
    return [v for v in synthetic_verification if v.get("outcome") == "passing"]


@pytest.fixture
def failing_verification(synthetic_verification: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to failing verification evidence."""
    return [v for v in synthetic_verification if v.get("outcome") == "failing"]

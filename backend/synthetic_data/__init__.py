"""Synthetic data generation for regulatory testing.

This package provides generators for:
- Scenarios: Test cases covering ontology dimensions and edge cases
- Rules: YAML rule definitions for expanded regulatory coverage
- Verification: Evidence records across consistency tiers

Usage:
    from backend.synthetic_data import ScenarioGenerator, RuleGenerator

    scenarios = ScenarioGenerator(seed=42).generate(count=500)
    rules = RuleGenerator(seed=42).generate(count=50)
"""

from backend.synthetic_data.base import BaseGenerator
from backend.synthetic_data.config import (
    THRESHOLDS,
    SCENARIO_CATEGORIES,
    RULE_DISTRIBUTIONS,
    VERIFICATION_TIERS,
    CONFIDENCE_RANGES,
)
from backend.synthetic_data.scenario_generator import ScenarioGenerator
from backend.synthetic_data.rule_generator import RuleGenerator
from backend.synthetic_data.verification_generator import VerificationGenerator

__all__ = [
    # Base
    "BaseGenerator",
    # Config
    "THRESHOLDS",
    "SCENARIO_CATEGORIES",
    "RULE_DISTRIBUTIONS",
    "VERIFICATION_TIERS",
    "CONFIDENCE_RANGES",
    # Generators
    "ScenarioGenerator",
    "RuleGenerator",
    "VerificationGenerator",
]

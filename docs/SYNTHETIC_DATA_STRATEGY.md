# Synthetic Data Strategy for RWA Regulatory Platform

**Review Date:** 2026-01-12
**Reviewer:** Claude (Opus 4.5)
**Branch:** `claude/review-synthetic-data-strategy-d1A9S`

---

## 1. Codebase Review Summary

### 1.1 Project Overview

This is a **Regulatory Knowledge Engineering Workbench** for Real-World Assets (RWAs) - a computational law platform that transforms regulatory documents into executable rules with traceable decision logic.

### 1.2 Supported Regulatory Frameworks

| Framework | Jurisdiction | Current Rules | Status |
|-----------|--------------|---------------|--------|
| MiCA 2023 | EU | ~9 rules | Production |
| FCA Crypto 2024 | UK | ~5 rules | Production |
| GENIUS Act | US | ~6 rules | Illustrative |
| RWA Tokenization | EU | ~3 rules | Illustrative |

### 1.3 Architecture Components

```
├── Core Engine
│   ├── Rule DSL Compiler (YAML → IR)
│   ├── Decision Engine (scenario evaluation)
│   ├── Jurisdiction Resolver (multi-region)
│   └── Temporal Engine (versioning)
│
├── ML Workflows
│   ├── Embedding Service (4-type vectors)
│   ├── Verification Service (5-tier checks)
│   ├── RAG Service (retrieval)
│   └── Decoder Service (explanations)
│
├── Persistence
│   ├── SQLite (default) / PostgreSQL (production)
│   ├── Event Sourcing (audit trail)
│   └── Embedding/Graph Stores
│
└── APIs (FastAPI)
    ├── /decide - Rule evaluation
    ├── /verify - Consistency checking
    ├── /search - Semantic search
    ├── /navigate - Cross-border compliance
    ├── /decoder - Explanations
    └── /counterfactual - What-if analysis
```

### 1.4 Core Data Models

**Ontological Types** (`backend/core/ontology/types.py`):
- **Actor Types:** issuer, offeror, trading_platform, custodian, investor, competent_authority
- **Instrument Types:** art, emt, stablecoin, utility_token, security_token, rwa_token, rwa_debt, rwa_equity, rwa_property
- **Activity Types:** public_offer, admission_to_trading, custody, tokenization, disclosure, valuation
- **Provision Types:** definition, scope, requirement, prohibition, exception, procedure, sanction

**Scenario Model** (`backend/core/ontology/scenario.py`):
- Flexible input model for decision queries
- Fields: instrument_type, activity, jurisdiction, authorized, actor_type, is_significant, reserve_value_eur, etc.

**Persistence Models** (`backend/core/models.py`):
- RuleRecord, RuleVersionRecord, RuleEventRecord
- VerificationResultRecord, VerificationEvidenceRecord
- ReviewRecord, PremiseIndexRecord

---

## 2. Current Data State

### 2.1 Rule Files

| File | Size | Rules | Framework |
|------|------|-------|-----------|
| `mica_authorization.yaml` | 3.5 KB | 2 | MiCA |
| `mica_stablecoin.yaml` | 13 KB | ~5 | MiCA |
| `rwa_authorization.yaml` | 5.4 KB | 3 | RWA |
| `fca_crypto.yaml` | 9 KB | ~5 | FCA |
| `genius_stablecoin.yaml` | 13 KB | ~6 | GENIUS |

**Total:** ~24 rules, 46 KB

### 2.2 Test Coverage

- **Test Files:** ~28 modules
- **Lines of Test Code:** ~6,917 lines
- **Test Types:** Persistence, corpus loading, compiler, navigation, RWA rules, consistency API

### 2.3 Gaps Identified

1. **Limited scenario coverage** - ~30 test cases for 24 rules
2. **No systematic edge case testing** - Threshold boundaries not covered
3. **Minimal cross-jurisdiction scenarios** - Limited multi-region testing
4. **No negative test cases** - Failure paths underrepresented
5. **Temporal testing gaps** - Version effectiveness not fully tested

---

## 3. Synthetic Data Strategy

### 3.1 Data Domains & Volumes

| Domain | Current | Target | Priority |
|--------|---------|--------|----------|
| Rules (YAML) | ~24 | 75-100 | **High** |
| Test Scenarios | ~30 | 400-600 | **High** |
| Embeddings | Auto | Auto | Medium |
| Verification Evidence | Minimal | 200-400 | Medium |
| Jurisdiction Data | 5 | 10+ regimes | Medium |
| Graph Relationships | Dynamic | 500-1000 | Low |

### 3.2 Rule Generation Strategy

#### Distribution by Framework
```
Target: 50-75 new rules
├── MiCA (EU): 15-20 rules
│   ├── Authorization (Art. 16, 36, 48)
│   ├── Stablecoin reserves (Art. 36-38)
│   ├── Custody requirements (Art. 75)
│   └── Disclosure obligations (Art. 19, 27)
│
├── FCA Crypto (UK): 12-15 rules
│   ├── Financial promotions
│   ├── Custody regime
│   └── Authorization requirements
│
├── GENIUS Act (US): 10-15 rules
│   ├── Stablecoin licensing
│   ├── Reserve requirements
│   └── Redemption rights
│
└── RWA Tokenization: 10-15 rules
    ├── Debt tokenization
    ├── Equity tokenization
    └── Property tokenization
```

#### Complexity Distribution
- **Simple (30%):** Single condition → single outcome
- **Medium (50%):** 2-3 nested conditions with alternatives
- **Complex (20%):** Multi-branch trees with cross-references

#### Rule Template Structure
```yaml
rule_id: "{framework}_{article}_{activity}"
version: "1.0"
effective_from: "YYYY-MM-DD"
tags: [framework, category, instrument_type, activity]

applies_if:
  all:
    - field: instrument_type
      operator: in
      value: [...]
    - field: activity
      operator: "=="
      value: ...

decision_tree:
  node_id: root
  condition:
    field: ...
    operator: ...
    value: ...
  true_branch: ...
  false_branch: ...

source:
  document_id: ...
  article: "..."
  pages: [...]
```

### 3.3 Scenario Generation Strategy

#### Dimensional Matrix
```python
SCENARIO_DIMENSIONS = {
    "instrument_type": [
        "art", "emt", "stablecoin", "utility_token",
        "security_token", "rwa_token", "rwa_debt", "rwa_equity"
    ],
    "activity": [
        "public_offer", "admission_to_trading", "custody",
        "tokenization", "disclosure", "exchange", "transfer"
    ],
    "jurisdiction": ["EU", "UK", "US", "CH", "SG"],
    "actor_type": [
        "issuer", "offeror", "trading_platform",
        "custodian", "investor"
    ],
    "authorized": [True, False],
    "is_significant": [True, False],
    "is_credit_institution": [True, False]
}
```

#### Scenario Categories

| Category | Target Count | Description |
|----------|--------------|-------------|
| Happy Path | 150 | Valid compliant scenarios |
| Edge Cases | 150 | Boundary conditions |
| Negative Cases | 100 | Rule violations |
| Cross-Border | 75 | Multi-jurisdiction |
| Temporal | 50 | Version-dependent |
| Counterfactual | 75 | What-if comparisons |

#### Threshold Testing Values
```yaml
# Reserve value thresholds
reserve_value_eur:
  - 4_999_999    # Below significant (5M)
  - 5_000_000    # At significant threshold
  - 5_000_001    # Above significant
  - 99_999_999   # Below large issuer (100M)
  - 100_000_000  # At large issuer threshold

# Token value thresholds
total_token_value_eur:
  - 999_999      # Below small issuer (1M)
  - 1_000_000    # At small issuer threshold
  - 4_999_999    # Below medium (5M)
  - 5_000_000    # At medium threshold
```

### 3.4 Embedding Generation

#### 4-Type Embeddings
```python
EMBEDDING_TYPES = {
    "semantic": {
        "source": "name + description + decision_explanation",
        "purpose": "Natural language search"
    },
    "structural": {
        "source": "conditions + operators + tree_structure",
        "purpose": "Structurally similar rules"
    },
    "entity": {
        "source": "field_names + operators",
        "purpose": "Same data field usage"
    },
    "legal": {
        "source": "citations + document_id + article",
        "purpose": "Same legal source"
    }
}

CONFIG = {
    "model": "all-MiniLM-L6-v2",
    "dimension": 384
}
```

#### Similarity Targets
- Same framework rules: cosine similarity 0.7-0.9
- Cross-framework similar: cosine similarity 0.5-0.7
- Unrelated rules: cosine similarity 0.1-0.3

### 3.5 Verification Evidence Strategy

#### 5-Tier Distribution
```yaml
Tier 0 (Schema):     40% - JSON schema validation
Tier 1 (Semantic):   25% - NLI consistency checks
Tier 2 (Cross-Rule): 15% - Inter-rule contradiction detection
Tier 3 (Temporal):   10% - Version consistency
Tier 4 (External):   10% - Source document alignment
```

#### Confidence Score Distribution
- Passing evidence: 0.85-0.99
- Marginal evidence: 0.70-0.84
- Failing evidence: 0.40-0.69

### 3.6 Jurisdiction & Equivalence Data

#### Equivalence Matrix
```
          EU    UK    US    CH    SG
EU        -     P     N     P     P
UK        P     -     N     P     P
US        N     N     -     N     N
CH        P     P     N     -     P
SG        P     P     N     P     -

Legend: P=Partial, N=Not Equivalent
```

#### Cross-Border Test Scenarios
1. EU issuer → UK target (post-Brexit)
2. UK issuer → EU target (equivalence check)
3. US issuer → EU target (strict requirements)
4. CH issuer → EU target (bilateral agreement)
5. SG issuer → UK target (fintech corridor)

---

## 4. Implementation Plan

### 4.1 Directory Structure

```
/backend/synthetic_data/
├── __init__.py
├── config.yaml
├── generators/
│   ├── __init__.py
│   ├── base.py
│   ├── rule_generator.py
│   ├── scenario_generator.py
│   ├── embedding_generator.py
│   ├── verification_generator.py
│   └── jurisdiction_generator.py
├── templates/
│   ├── rule_templates.yaml
│   ├── scenario_templates.yaml
│   └── evidence_templates.yaml
├── validators/
│   ├── __init__.py
│   ├── rule_validator.py
│   └── scenario_validator.py
└── output/
    ├── rules/
    ├── scenarios/
    ├── embeddings/
    └── fixtures/
```

### 4.2 Generator Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random

class BaseGenerator(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def generate(self, count: int) -> List[Dict[str, Any]]:
        """Generate synthetic data items."""
        pass

    @abstractmethod
    def validate(self, item: Dict[str, Any]) -> bool:
        """Validate generated item against schema."""
        pass

    def generate_batch(self, count: int) -> List[Dict[str, Any]]:
        """Generate and validate a batch of items."""
        items = []
        while len(items) < count:
            item = self.generate(1)[0]
            if self.validate(item):
                items.append(item)
        return items
```

### 4.3 Pytest Integration

```python
import pytest
from backend.synthetic_data import SyntheticDataLoader

@pytest.fixture(scope="session")
def synthetic_rules():
    """Load synthetic rules for testing."""
    return SyntheticDataLoader.load_rules(count=50)

@pytest.fixture(scope="session")
def synthetic_scenarios():
    """Load synthetic scenarios for testing."""
    return SyntheticDataLoader.load_scenarios(count=300)

@pytest.fixture
def synthetic_scenario_matrix():
    """Generate scenario matrix for parametrized tests."""
    return SyntheticDataLoader.generate_matrix(
        instrument_types=["art", "emt", "stablecoin"],
        activities=["public_offer", "custody"],
        jurisdictions=["EU", "UK"]
    )
```

### 4.4 Phased Rollout

| Phase | Deliverable | Timeline | Dependencies |
|-------|-------------|----------|--------------|
| **1** | Scenario Generator | First | Ontology types |
| **2** | Rule Generator | After P1 | Scenario generator |
| **3** | Verification Evidence | After P2 | Rules + scenarios |
| **4** | Embedding Pipeline | After P2 | Rules |
| **5** | Graph Relationships | After P4 | Embeddings |

---

## 5. Quality Assurance

### 5.1 Validation Checklist

- [ ] All generated rules pass YAML schema validation
- [ ] All generated rules compile to valid IR
- [ ] Scenarios cover 100% of ontology enum values
- [ ] Scenarios include all threshold boundary values
- [ ] Embeddings cluster correctly by semantic similarity
- [ ] Cross-references resolve to valid entities
- [ ] Temporal dates are logically consistent
- [ ] Source citations reference valid articles

### 5.2 Coverage Metrics

| Metric | Target |
|--------|--------|
| Instrument type coverage | 100% |
| Activity coverage | 100% |
| Jurisdiction coverage | 100% |
| Actor type coverage | 100% |
| Decision outcome coverage | ≥90% |
| Edge case coverage | ≥80% |
| Cross-border scenario coverage | ≥70% |

### 5.3 Regression Testing

After synthetic data generation, run:
```bash
# Full test suite with synthetic data
pytest tests/ --synthetic-data

# Coverage report
pytest tests/ --cov=backend --cov-report=html

# Specific synthetic data tests
pytest tests/synthetic/ -v
```

---

## 6. Dependencies

### 6.1 Required Libraries

```
faker>=18.0.0          # Realistic data generation
sentence-transformers  # Embedding generation (existing)
pyyaml                 # YAML rule generation (existing)
numpy                  # Vector operations (existing)
networkx               # Graph generation
pydantic               # Schema validation (existing)
```

### 6.2 Existing Code Integration

The synthetic data generators should integrate with:
- `backend/core/ontology/types.py` - Use existing enums
- `backend/core/ontology/scenario.py` - Match Scenario model
- `backend/core/models.py` - Match persistence models
- `backend/rule_service/compiler.py` - Validate rule IR compilation

---

## 7. Appendix

### A. Sample Synthetic Rule

```yaml
rule_id: mica_art48_emt_authorization
version: "1.0"
effective_from: "2024-06-30"
jurisdiction: EU
tags: [mica, authorization, emt, e-money]

applies_if:
  all:
    - field: instrument_type
      operator: "=="
      value: emt
    - field: activity
      operator: in
      value: [public_offer, admission_to_trading]
    - field: jurisdiction
      operator: "=="
      value: EU

decision_tree:
  node_id: check_authorization
  condition:
    field: is_credit_institution
    operator: "=="
    value: true
  true_branch:
    result: authorized
    explanation: "Credit institutions are authorized to issue EMTs under existing banking license"
  false_branch:
    node_id: check_emi_license
    condition:
      field: is_emi
      operator: "=="
      value: true
    true_branch:
      result: authorized
      explanation: "E-money institutions may issue EMTs under MiCA Art. 48"
    false_branch:
      result: not_authorized
      obligations:
        - id: obtain_emi_license
          description: "Obtain e-money institution authorization per Directive 2009/110/EC"

source:
  document_id: mica_2023
  article: "48(1)"
  pages: [82, 83]
```

### B. Sample Synthetic Scenario

```yaml
scenario_id: syn_mica_emt_credit_institution_001
description: "Credit institution issuing EMT in EU - should be authorized"
category: happy_path

input:
  instrument_type: emt
  activity: public_offer
  jurisdiction: EU
  actor_type: issuer
  authorized: false
  is_credit_institution: true
  is_emi: false
  is_significant: false

expected:
  applicable_rules: [mica_art48_emt_authorization]
  decision: authorized
  obligations: []
```

### C. Sample Verification Evidence

```yaml
evidence_id: ver_001
rule_id: mica_art48_emt_authorization
tier: 1
category: semantic_consistency
label: "Decision tree logic consistency"
score: 0.92
details:
  check: "NLI entailment between condition and outcome"
  finding: "True branch correctly identifies credit institution exemption"
  source_span: "lines 18-22"
  rule_element: "decision_tree.true_branch"
```

---

*Document generated as part of codebase review for synthetic data strategy planning.*

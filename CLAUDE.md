# Claude Code Project: Regulatory KE Workbench

## Overview

A **computational law system** for tokenized real-world assets (RWAs). The system provides executable legal logic, traceable decisions, and multi-jurisdiction compliance analysis.

## Project Structure

```
droit/
├── backend/
│   ├── core/                           # Shared components
│   │   ├── config.py                   # Application settings
│   │   ├── database.py                 # SQLModel engine/session
│   │   ├── api/                        # FastAPI routes
│   │   │   ├── main.py                 # App entry point
│   │   │   ├── routes_decide.py        # /decide endpoint
│   │   │   ├── routes_ke.py            # KE dashboard endpoints
│   │   │   └── routes_production.py    # Production API
│   │   ├── ontology/                   # Domain types
│   │   │   ├── types.py                # Provision, Actor, Instrument
│   │   │   ├── scenario.py             # Scenario model
│   │   │   └── jurisdiction.py         # JurisdictionCode, RuleConflict
│   │   └── visualization/              # Tree rendering adapters
│   │
│   ├── rule_service/                   # Rule management + evaluation
│   │   ├── app/services/
│   │   │   ├── loader.py               # RuleLoader, Rule, SourceRef
│   │   │   ├── engine.py               # DecisionEngine, TraceStep
│   │   │   ├── schema.py               # ConsistencyBlock, ConsistencyEvidence
│   │   │   └── jurisdiction/           # Multi-jurisdiction navigation
│   │   │       ├── resolver.py         # resolve_jurisdictions
│   │   │       ├── evaluator.py        # evaluate_jurisdiction
│   │   │       ├── conflicts.py        # detect_conflicts
│   │   │       └── pathway.py          # synthesize_pathway
│   │   └── data/                       # YAML rule packs
│   │       ├── mica_authorization.yaml
│   │       ├── mica_stablecoin.yaml
│   │       ├── fca_crypto.yaml
│   │       └── rwa_authorization.yaml
│   │
│   ├── database_service/               # Data access middleware
│   │   └── app/services/
│   │       ├── database.py             # init_db, seed operations
│   │       ├── migration.py            # YAML to DB migration
│   │       ├── compiler/               # Rule IR compilation
│   │       │   ├── ir.py               # RuleIR, CompiledCheck
│   │       │   ├── compiler.py         # RuleCompiler
│   │       │   └── premise_index.py    # O(1) premise lookup
│   │       ├── runtime/                # IR execution
│   │       │   ├── executor.py         # RuleRuntime
│   │       │   └── cache.py            # IRCache
│   │       └── repositories/           # Database CRUD
│   │
│   ├── verification_service/           # Consistency engine
│   │   └── app/services/
│   │       └── consistency_engine.py   # ConsistencyEngine (Tier 0-4)
│   │
│   ├── analytics_service/              # Error analysis + drift
│   │   └── app/services/
│   │       ├── error_patterns.py       # ErrorPatternAnalyzer
│   │       └── drift.py                # DriftDetector
│   │
│   ├── rag_service/                    # Retrieval + Q&A
│   │   └── app/services/
│   │       ├── retrieval.py            # Retriever, BM25Index
│   │       ├── corpus_loader.py        # Legal document loading
│   │       ├── rule_context.py         # RuleContextRetriever
│   │       └── frontend_helpers.py     # UI helper functions
│   │
│   ├── rule_embedding_service/         # Rule embeddings
│   │   └── app/services/
│   │       └── embedding.py            # Embedding operations
│   │
│   └── main.py                         # Application entry
│
├── frontend/                           # Streamlit UI
│   ├── Home.py                         # Landing page
│   ├── pages/
│   │   ├── 1_KE_Workbench.py          # Rule editor + verification
│   │   ├── 2_Production_Demo.py       # Production deployment demo
│   │   └── 3_Navigator.py             # Cross-border navigator
│   └── ui/                             # Shared UI components
│
├── data/
│   └── legal/                          # Legal corpus (MiCA, DLT Pilot, etc.)
│
├── docs/                               # Design documentation
│   ├── engine_design.md
│   ├── rule_dsl.md
│   └── knowledge_model.md
│
└── tests/                              # Test suite (474 tests)
```

## Key Concepts

### Rule Model
Rules are stored in YAML files under `backend/rule_service/data/`. Each rule has:
- `rule_id`: Unique identifier (e.g., `mica_art36_public_offer_authorization`)
- `source`: Legal source reference (document_id, article)
- `jurisdiction`: JurisdictionCode (EU, UK, US, CH, SG)
- `decision_tree`: Nested conditions with outcomes
- `effective_date`: When the rule takes effect

### Consistency Engine (Tier 0-4)
- **Tier 0**: Schema validation (required fields, types)
- **Tier 1**: Lexical checks (dates, ID format, tags)
- **Tier 2**: Semantic similarity (rule vs source text)
- **Tier 3**: NLI entailment checks
- **Tier 4**: Cross-rule consistency (conflict detection)

### Cross-Border Navigation
The jurisdiction module handles multi-jurisdiction compliance:
1. `resolve_jurisdictions`: Identify applicable regimes
2. `evaluate_jurisdiction`: Assess compliance per jurisdiction
3. `detect_conflicts`: Find cross-border rule conflicts
4. `synthesize_pathway`: Generate compliance pathway

## Development Commands

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Run tests
pytest

# Run specific test file
pytest tests/test_rules.py -v

# Run API server
uvicorn backend.core.api.main:app --reload

# Run Streamlit dashboard
streamlit run frontend/Home.py
```

## Import Patterns

```python
# Core ontology types
from backend.core.ontology import Scenario, JurisdictionCode

# Rule service
from backend.rule_service.app.services import RuleLoader, DecisionEngine, Rule

# Verification
from backend.verification_service.app.services import ConsistencyEngine

# Database operations
from backend.database_service.app.services import init_db_with_seed, migrate_yaml_rules

# RAG and retrieval
from backend.rag_service.app.services import Retriever, BM25Index

# Analytics
from backend.analytics_service.app.services import ErrorPatternAnalyzer, DriftDetector
```

## Guidelines

1. **Rule files**: Keep legal logic in YAML under `backend/rule_service/data/`, not hard-coded in Python
2. **Testing**: Run `pytest` after changes, target 474+ tests passing
3. **Imports**: Use the canonical service paths shown above
4. **Documentation**: Update `docs/*.md` when changing ontology or engine behavior
5. **Incremental changes**: Prefer small, focused changes over large refactors

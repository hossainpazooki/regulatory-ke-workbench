# Migration Plan: Flat Domain Structure

## Overview

Restructure the backend from nested `*_service` directories to flat domain directories following FastAPI best practices: **"Organize code by domain, not by file type."**

---

## Current API Endpoints Reference

### Complete Endpoint Inventory

| Prefix | Tag | Current Source | Endpoint Count |
|--------|-----|----------------|----------------|
| `/qa` | Q&A | `routes_qa.py` | 3 |
| `/decide` | Decisions | `routes_decide.py` | 2 |
| `/rules` | Rules | `routes_rules.py` | 7 |
| `/ke` | Knowledge Engineering | `routes_ke.py` | 22 |
| `/v2` | Production API | `routes_production.py` | 12 |
| `/navigate` | Navigate | `routes_navigate.py` | 4 |
| `/decoder` | Decoder | `routes_decoder.py` | 5 |
| `/counterfactual` | Counterfactual | `routes_counterfactual.py` | 5 |
| `/analytics` | Analytics | `routes_analytics.py` | 11 |
| `/risk` | Market Risk | `routes_risk.py` | 4 |
| `/embedding/rules` | Embeddings | `embedding_service/routes.py` | varies |
| `/embedding/search` | Search | `embedding_service/search.py` | varies |
| `/embedding/graph` | Graph | `embedding_service/graph.py` | varies |

---

## Detailed Endpoint Mapping

### 1. RAG Domain (`/qa`)
**Current:** `core/api/routes_qa.py`
**Target:** `rag/router.py`

```
POST /qa/ask          # Ask a question
POST /qa/index        # Index documents
GET  /qa/status       # Get indexing status
```

### 2. Rules Domain (`/decide`, `/rules`)
**Current:** `core/api/routes_decide.py`, `core/api/routes_rules.py`
**Target:** `rules/router.py`

```
# Decision endpoints
POST /decide          # Evaluate rules against scenario
POST /decide/reload   # Reload rules from YAML

# Rule inspection endpoints
GET  /rules           # List all rules
GET  /rules/{rule_id} # Get rule details
GET  /rules/tags/all  # Get all tags
GET  /rules/{rule_id}/versions           # List rule versions
GET  /rules/{rule_id}/versions/{version} # Get specific version
GET  /rules/{rule_id}/at-timestamp       # Get rule at timestamp
GET  /rules/{rule_id}/events             # Get rule events
```

### 3. Knowledge Engineering Domain (`/ke`)
**Current:** `core/api/routes_ke.py`
**Target:** `ke/router.py`

```
# Verification
POST /ke/verify                    # Verify single rule
POST /ke/verify-all                # Verify all rules

# Analytics
GET  /ke/analytics/summary         # Analytics summary
GET  /ke/analytics/patterns        # Error patterns
GET  /ke/analytics/matrix          # Error matrix
GET  /ke/analytics/review-queue    # Review queue

# Drift detection
POST /ke/drift/baseline            # Set baseline
GET  /ke/drift/detect              # Detect drift
GET  /ke/drift/history             # Drift history
GET  /ke/drift/authors             # Author performance

# Context
GET  /ke/context/{rule_id}         # Get rule context
GET  /ke/related/{rule_id}         # Get related rules

# Human review
POST /ke/rules/{rule_id}/review    # Submit review
GET  /ke/rules/{rule_id}/reviews   # Get reviews

# Charts
GET  /ke/charts/supertree-status         # Supertree status
GET  /ke/charts/rulebook-outline         # Rulebook outline
GET  /ke/charts/rulebook-outline/html    # Rulebook outline HTML
GET  /ke/charts/ontology                 # Ontology chart
GET  /ke/charts/ontology/html            # Ontology HTML
GET  /ke/charts/corpus-links             # Corpus links
GET  /ke/charts/corpus-links/html        # Corpus links HTML
GET  /ke/charts/decision-tree/{rule_id}  # Decision tree
POST /ke/charts/decision-trace/{rule_id}      # Decision trace
POST /ke/charts/decision-trace/{rule_id}/html # Decision trace HTML
```

### 4. Production API Domain (`/v2`)
**Current:** `core/api/routes_production.py`
**Target:** `production/router.py`

```
# Migration
POST /v2/migrate                   # Migrate YAML to DB

# Status
GET  /v2/status                    # Database stats

# Compilation
POST /v2/rules/{rule_id}/compile   # Compile single rule
POST /v2/rules/compile             # Compile all rules

# Evaluation
POST /v2/rules/{rule_id}/evaluate  # Evaluate single rule
POST /v2/evaluate                  # Batch evaluate

# Cache management
GET  /v2/cache/stats               # Cache statistics
POST /v2/cache/clear               # Clear cache
POST /v2/cache/preload             # Preload cache

# Index management
GET  /v2/index/stats               # Index statistics
POST /v2/index/rebuild             # Rebuild index
GET  /v2/index/lookup              # Lookup by premise
```

### 5. Jurisdiction Domain (`/navigate`)
**Current:** `core/api/routes_navigate.py`
**Target:** `jurisdiction/router.py`

```
POST /navigate                     # Cross-border navigation
GET  /navigate/jurisdictions       # List jurisdictions
GET  /navigate/regimes             # List regulatory regimes
GET  /navigate/equivalences        # List equivalence mappings
```

### 6. Decoder Domain (`/decoder`, `/counterfactual`)
**Current:** `core/api/routes_decoder.py`, `core/api/routes_counterfactual.py`
**Target:** `decoder/router.py`

```
# Explanation endpoints
POST /decoder/explain              # Generate explanation
POST /decoder/explain/inline       # Inline explanation
GET  /decoder/templates            # List templates
GET  /decoder/templates/{id}       # Get template
GET  /decoder/tiers                # List explanation tiers

# Counterfactual endpoints
POST /counterfactual/analyze           # Analyze scenario
POST /counterfactual/analyze/inline    # Inline analysis
POST /counterfactual/compare           # Compare scenarios
POST /counterfactual/compare/inline    # Inline comparison
GET  /counterfactual/scenario-types    # List scenario types
```

### 7. Analytics Domain (`/analytics`)
**Current:** `core/api/routes_analytics.py`
**Target:** `analytics/router.py`

```
POST /analytics/rules/compare      # Compare rules
GET  /analytics/rule-clusters      # Get clusters
POST /analytics/rule-clusters      # Generate clusters
POST /analytics/find-conflicts     # Find conflicts
GET  /analytics/conflicts          # Get conflicts
GET  /analytics/rules/{rule_id}/similar  # Find similar rules
POST /analytics/rules/similar      # Find similar (custom params)
GET  /analytics/coverage           # Coverage report
GET  /analytics/umap-projection    # UMAP projection
POST /analytics/umap-projection    # UMAP (custom params)
GET  /analytics/summary            # Analytics summary
```

### 8. Market Risk Domain (`/risk`)
**Current:** `core/api/routes_risk.py`
**Target:** `market_risk/router.py`

```
POST /risk/assess                          # Assess position risk
GET  /risk/market-intelligence/{asset}     # Market intelligence
POST /risk/calculate-var                   # Calculate VaR
GET  /risk/supported-assets                # List supported assets
```

### 9. Embeddings Domain (`/embedding/*`)
**Current:** `rule_embedding_service/app/services/routes.py`, `*/api/routes/*.py`
**Target:** `embeddings/router.py`

```
# Rule embeddings (prefix: /embedding/rules)
# Current source: rule_embedding_service/app/services/routes.py

# Search (prefix: /embedding/search)
# Current source: rule_embedding_service/app/api/routes/search.py

# Graph (prefix: /embedding/graph)
# Current source: rule_embedding_service/app/api/routes/graph.py

# Base embeddings (prefix: /embedding)
# Current source: rule_embedding_service/app/api/routes/embeddings.py
```

### 10. NEW: DeFi Risk Domain (`/defi-risk`)
**Current:** `rule_service/app/services/defi_risk/scoring.py` (no router)
**Target:** `defi_risk/router.py`

```
POST /defi-risk/score              # Score a DeFi protocol
GET  /defi-risk/protocols          # List protocol defaults
GET  /defi-risk/protocols/{id}     # Get protocol config
GET  /defi-risk/categories         # List DeFi categories
```

### 11. NEW: Token Compliance Domain (`/token-compliance`)
**Current:** `rule_service/app/services/token_standards/compliance.py` (no router)
**Target:** `token_compliance/router.py`

```
POST /token-compliance/howey-test      # Apply Howey test
POST /token-compliance/genius-act      # GENIUS Act analysis
POST /token-compliance/analyze         # Full compliance analysis
GET  /token-compliance/standards       # List token standards
```

### 12. NEW: Protocol Risk Domain (`/protocol-risk`)
**Current:** `rule_service/app/services/protocol_risk/consensus.py` (no router)
**Target:** `protocol_risk/router.py`

```
POST /protocol-risk/assess             # Assess protocol risk
GET  /protocol-risk/consensus-types    # List consensus types
```

---

## Current vs Target Structure

### Current Structure
```
backend/
├── main.py
├── config.py
├── core/
│   ├── api/
│   │   ├── __init__.py           # Exports all routers
│   │   ├── routes_qa.py          # /qa
│   │   ├── routes_decide.py      # /decide
│   │   ├── routes_rules.py       # /rules
│   │   ├── routes_ke.py          # /ke
│   │   ├── routes_production.py  # /v2
│   │   ├── routes_navigate.py    # /navigate
│   │   ├── routes_decoder.py     # /decoder
│   │   ├── routes_counterfactual.py  # /counterfactual
│   │   ├── routes_analytics.py   # /analytics
│   │   └── routes_risk.py        # /risk
│   ├── models.py
│   ├── database.py
│   └── ontology/
├── rule_service/
│   └── app/services/
│       ├── engine.py
│       ├── loader.py
│       ├── defi_risk/scoring.py      # No router (problem!)
│       ├── market_risk/volatility.py # Used by routes_risk.py
│       ├── protocol_risk/consensus.py # No router (problem!)
│       ├── token_standards/compliance.py # No router (problem!)
│       └── jurisdiction/
├── analytics_service/
├── decoder_service/
├── database_service/
├── rag_service/
└── rule_embedding_service/
```

### Target Structure
```
backend/
├── main.py
├── config.py
├── database.py
│
├── rules/                    # /decide, /rules
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── service.py            # DecisionEngine
│   ├── loader.py             # RuleLoader
│   ├── versioning.py
│   └── constants.py
│
├── jurisdiction/             # /navigate
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── service.py            # JurisdictionResolver
│   ├── evaluator.py
│   ├── conflicts.py
│   ├── pathway.py
│   └── constants.py
│
├── defi_risk/                # /defi-risk (NEW)
│   ├── __init__.py
│   ├── router.py             # NEW
│   ├── schemas.py
│   ├── service.py
│   └── constants.py
│
├── token_compliance/         # /token-compliance (NEW)
│   ├── __init__.py
│   ├── router.py             # NEW
│   ├── schemas.py
│   ├── service.py
│   └── constants.py
│
├── market_risk/              # /risk
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── service.py
│   └── constants.py
│
├── protocol_risk/            # /protocol-risk (NEW)
│   ├── __init__.py
│   ├── router.py             # NEW
│   ├── schemas.py
│   └── service.py
│
├── analytics/                # /analytics
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── service.py
│   ├── clustering.py
│   ├── drift.py
│   └── error_patterns.py
│
├── decoder/                  # /decoder, /counterfactual
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── service.py
│   ├── counterfactual.py
│   ├── templates.py
│   ├── citations.py
│   └── delta.py
│
├── embeddings/               # /embedding/*
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── models.py
│   ├── service.py
│   ├── generator.py
│   └── graph.py
│
├── rag/                      # /qa
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   ├── service.py
│   ├── bm25.py
│   ├── chunker.py
│   └── corpus_loader.py
│
├── production/               # /v2
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py
│   └── service.py
│
├── ke/                       # /ke
│   ├── __init__.py
│   ├── router.py
│   └── schemas.py
│
├── storage/                  # Shared data persistence
│   ├── __init__.py
│   ├── database.py
│   ├── models.py
│   ├── repositories/
│   ├── stores/
│   └── temporal/
│
└── shared/                   # Shared utilities
    ├── __init__.py
    ├── exceptions.py
    ├── ontology/
    └── visualization/
```

---

## Updated main.py

```python
"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.storage.database import init_db

# Import routers from each domain
from backend.rag.router import router as rag_router
from backend.rules.router import decide_router, rules_router
from backend.ke.router import router as ke_router
from backend.production.router import router as production_router
from backend.jurisdiction.router import router as jurisdiction_router
from backend.decoder.router import decoder_router, counterfactual_router
from backend.analytics.router import router as analytics_router
from backend.market_risk.router import router as market_risk_router
from backend.defi_risk.router import router as defi_risk_router
from backend.token_compliance.router import router as token_compliance_router
from backend.protocol_risk.router import router as protocol_risk_router
from backend.embeddings.router import (
    rules_router as embedding_rules_router,
    search_router as embedding_search_router,
    graph_router as embedding_graph_router,
    embeddings_router as embedding_base_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    print(f"Starting {settings.app_name}...")
    init_db()
    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Computational law platform for tokenized real-world assets",
        version="0.2.0",  # Version bump for restructure
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all domain routers
    # Existing endpoints (unchanged URLs)
    app.include_router(rag_router)               # /qa
    app.include_router(decide_router)            # /decide
    app.include_router(rules_router)             # /rules
    app.include_router(ke_router)                # /ke
    app.include_router(production_router)        # /v2
    app.include_router(jurisdiction_router)      # /navigate
    app.include_router(decoder_router)           # /decoder
    app.include_router(counterfactual_router)    # /counterfactual
    app.include_router(analytics_router)         # /analytics
    app.include_router(market_risk_router)       # /risk

    # Embedding routers (unchanged URLs)
    app.include_router(embedding_rules_router)   # /embedding/rules
    app.include_router(embedding_search_router)  # /embedding/search
    app.include_router(embedding_graph_router)   # /embedding/graph
    app.include_router(embedding_base_router)    # /embedding

    # NEW endpoints (new URLs)
    app.include_router(defi_risk_router)         # /defi-risk
    app.include_router(token_compliance_router)  # /token-compliance
    app.include_router(protocol_risk_router)     # /protocol-risk

    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": "0.2.0",
            "endpoints": {
                # Existing
                "qa": "/qa/ask",
                "decide": "/decide",
                "rules": "/rules",
                "ke": "/ke/*",
                "v2": "/v2/*",
                "navigate": "/navigate",
                "decoder": "/decoder/*",
                "counterfactual": "/counterfactual/*",
                "analytics": "/analytics/*",
                "risk": "/risk/*",
                "embedding": "/embedding/*",
                # New
                "defi-risk": "/defi-risk/*",
                "token-compliance": "/token-compliance/*",
                "protocol-risk": "/protocol-risk/*",
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


app = create_app()
```

---

## Migration Phases

### Phase 1: Create Directory Structure
```bash
cd backend
mkdir -p rules jurisdiction defi_risk token_compliance market_risk protocol_risk
mkdir -p analytics decoder embeddings rag production ke
mkdir -p storage/repositories storage/stores storage/temporal
mkdir -p shared/ontology shared/visualization
```

### Phase 2: Extract Risk Domains (New Routers)

#### 2.1 defi_risk/router.py (NEW)
```python
"""DeFi risk scoring endpoints."""
from fastapi import APIRouter
from . import service, schemas

router = APIRouter(prefix="/defi-risk", tags=["defi-risk"])

@router.post("/score", response_model=schemas.DeFiRiskScore)
async def score_protocol(request: schemas.DeFiScoreRequest):
    """Score a DeFi protocol across risk dimensions."""
    return service.score_defi_protocol(
        protocol_id=request.protocol_id,
        category=request.category,
        smart_contract=request.smart_contract,
        economic=request.economic,
        oracle=request.oracle,
        governance=request.governance,
    )

@router.get("/protocols")
async def list_protocol_defaults():
    """List available protocol default configurations."""
    return {"protocols": list(service.DEFI_PROTOCOL_DEFAULTS.keys())}

@router.get("/protocols/{protocol_id}")
async def get_protocol_config(protocol_id: str):
    """Get default configuration for a known protocol."""
    config = service.DEFI_PROTOCOL_DEFAULTS.get(protocol_id)
    if not config:
        raise HTTPException(404, f"Protocol {protocol_id} not found")
    return config

@router.get("/categories")
async def list_categories():
    """List DeFi protocol categories."""
    return {"categories": [c.value for c in schemas.DeFiCategory]}
```

#### 2.2 token_compliance/router.py (NEW)
```python
"""Token compliance analysis endpoints."""
from fastapi import APIRouter
from . import service, schemas

router = APIRouter(prefix="/token-compliance", tags=["token-compliance"])

@router.post("/howey-test", response_model=schemas.HoweyTestResult)
async def apply_howey_test(request: schemas.HoweyTestRequest):
    """Apply SEC Howey Test to determine security classification."""
    return service.apply_howey_test(
        investment_of_money=request.investment_of_money,
        common_enterprise=request.common_enterprise,
        expectation_of_profits=request.expectation_of_profits,
        efforts_of_others=request.efforts_of_others,
        decentralization_score=request.decentralization_score,
        is_functional_network=request.is_functional_network,
    )

@router.post("/genius-act", response_model=schemas.GeniusActAnalysis)
async def analyze_genius_act(request: schemas.GeniusActRequest):
    """Analyze compliance with GENIUS Act stablecoin provisions."""
    return service.analyze_genius_act_compliance(
        is_stablecoin=request.is_stablecoin,
        pegged_currency=request.pegged_currency,
        reserve_assets=request.reserve_assets,
        reserve_ratio=request.reserve_ratio,
        uses_algorithmic_mechanism=request.uses_algorithmic_mechanism,
        issuer_charter_type=request.issuer_charter_type,
        has_reserve_attestation=request.has_reserve_attestation,
        attestation_frequency_days=request.attestation_frequency_days,
    )

@router.post("/analyze", response_model=schemas.TokenComplianceResult)
async def analyze_token_compliance(request: schemas.TokenComplianceRequest):
    """Comprehensive token compliance analysis."""
    return service.analyze_token_compliance(**request.model_dump())

@router.get("/standards")
async def list_token_standards():
    """List supported token standards."""
    return {"standards": [s.value for s in schemas.TokenStandard]}
```

### Phase 3: Move Existing Routes to Domains

Each domain's `router.py` inherits the exact same endpoints from the current `routes_*.py` files, just with updated imports.

**Example: market_risk/router.py**
```python
"""Market risk analytics endpoints."""
# This file is moved from core/api/routes_risk.py
# Only import paths change

from fastapi import APIRouter, HTTPException, Query
from . import service  # Changed from: backend.rule_service.app.services.market_risk
from .schemas import (  # Changed from inline definitions
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    # ... etc
)

router = APIRouter(prefix="/risk", tags=["market-risk"])

# All endpoints remain exactly the same
@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_position_risk(request: RiskAssessmentRequest):
    # ... existing implementation
```

### Phase 4: Update Cross-Domain Imports

```python
# In rules/service.py
from backend.defi_risk import service as defi_risk_service
from backend.token_compliance import service as token_service

# In jurisdiction/service.py
from backend.rules import service as rules_service

# In ke/router.py
from backend.analytics import service as analytics_service
from backend.rules import loader as rule_loader
```

### Phase 5: Update main.py Imports

Replace the old `core/api/__init__.py` imports with domain imports.

### Phase 6: Delete Old Structure

```bash
rm -rf backend/rule_service
rm -rf backend/analytics_service
rm -rf backend/decoder_service
rm -rf backend/database_service
rm -rf backend/rag_service
rm -rf backend/rule_embedding_service
rm -rf backend/verification_service
rm -rf backend/core/api/routes_*.py
```

---

## File Movement Reference

| Old Path | New Path |
|----------|----------|
| `core/api/routes_qa.py` | `rag/router.py` |
| `core/api/routes_decide.py` | `rules/router.py` |
| `core/api/routes_rules.py` | `rules/router.py` (merge) |
| `core/api/routes_ke.py` | `ke/router.py` |
| `core/api/routes_production.py` | `production/router.py` |
| `core/api/routes_navigate.py` | `jurisdiction/router.py` |
| `core/api/routes_decoder.py` | `decoder/router.py` |
| `core/api/routes_counterfactual.py` | `decoder/router.py` (merge) |
| `core/api/routes_analytics.py` | `analytics/router.py` |
| `core/api/routes_risk.py` | `market_risk/router.py` |
| `rule_service/app/services/engine.py` | `rules/service.py` |
| `rule_service/app/services/loader.py` | `rules/loader.py` |
| `rule_service/app/services/schema.py` | `rules/schemas.py` |
| `rule_service/app/services/versioning.py` | `rules/versioning.py` |
| `rule_service/app/services/jurisdiction/*` | `jurisdiction/*.py` |
| `rule_service/app/services/defi_risk/scoring.py` | `defi_risk/service.py` + `schemas.py` |
| `rule_service/app/services/market_risk/volatility.py` | `market_risk/service.py` + `schemas.py` |
| `rule_service/app/services/protocol_risk/consensus.py` | `protocol_risk/service.py` |
| `rule_service/app/services/token_standards/compliance.py` | `token_compliance/service.py` + `schemas.py` |
| `analytics_service/app/services/*` | `analytics/*.py` |
| `decoder_service/app/services/*` | `decoder/*.py` |
| `database_service/app/services/*` | `storage/*.py` |
| `rag_service/app/services/*` | `rag/*.py` |
| `rule_embedding_service/app/*` | `embeddings/*.py` |
| `core/ontology/*` | `shared/ontology/*` |
| `core/visualization/*` | `shared/visualization/*` |
| `core/models.py` | `storage/models.py` |
| `core/database.py` | `storage/database.py` |

---

## Import Change Reference

| Old Import | New Import |
|------------|------------|
| `from backend.rule_service.app.services.market_risk import *` | `from backend.market_risk import service, schemas` |
| `from backend.rule_service.app.services.defi_risk.scoring import *` | `from backend.defi_risk import service, schemas` |
| `from backend.rule_service.app.services.token_standards.compliance import *` | `from backend.token_compliance import service, schemas` |
| `from backend.rule_service.app.services.engine import DecisionEngine` | `from backend.rules.service import DecisionEngine` |
| `from backend.rule_service.app.services.loader import RuleLoader` | `from backend.rules.loader import RuleLoader` |
| `from backend.core.api import risk_router` | `from backend.market_risk.router import router` |
| `from backend.analytics_service.app.services.rule_analytics import *` | `from backend.analytics.service import *` |
| `from backend.decoder_service.app.services.decoder import *` | `from backend.decoder.service import *` |
| `from backend.core.ontology.scenario import Scenario` | `from backend.shared.ontology.scenario import Scenario` |
| `from backend.core.models import RuleRecord` | `from backend.storage.models import RuleRecord` |

---

## Testing Checklist

After each phase, verify:

```bash
# Lint check
ruff check backend/

# Type check (if using mypy)
mypy backend/

# Run all tests
pytest backend/tests/ -v

# API contract test - verify OpenAPI spec unchanged
python -c "from backend.main import app; print(app.openapi())"

# Manual endpoint test
uvicorn backend.main:app --reload
curl http://localhost:8000/docs  # Verify Swagger UI loads
```

---

## Deployment Notes

1. **Zero downtime**: All URL paths remain unchanged
2. **API contract**: OpenAPI spec should be identical
3. **Breaking changes**: None for existing endpoints
4. **New endpoints**: `/defi-risk/*`, `/token-compliance/*`, `/protocol-risk/*`
5. **Version bump**: 0.1.0 → 0.2.0 (new features, no breaks)

---

## Rollback Plan

If issues arise after partial migration:
1. Restore `core/api/__init__.py` to import from old locations
2. Old `*_service` directories remain functional
3. Revert `main.py` to previous version
4. Delete new domain directories

Keep old directories until Phase 6 completes successfully.

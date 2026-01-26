# Regulatory Knowledge Engineering Workbench

An open-source platform for encoding financial regulations as executable logic. Designed for MiCA, stablecoin frameworks, and RWA tokenization—with ML-powered explanations, cross-border compliance navigation, and interactive visualization tools.

**Live Demo:** [pazooki.streamlit.app](https://pazooki.streamlit.app)

## Applied AI Workbench

The Streamlit frontend provides seven interactive tools for exploring and analyzing regulatory rules:

| Page | Capability |
|------|------------|
| **KE Workbench** | Browse rules, visualize decision trees, run trace tests, verify consistency |
| **Production Demo** | Synthetic data strategy, deployment guardrails, O(1) premise lookup benchmarks |
| **Navigator** | Cross-border compliance pathways with conflict detection and equivalence mapping |
| **Embedding Explorer** | UMAP 2D/3D visualization of rule embeddings (semantic, structural, entity, legal) |
| **Similarity Search** | Find related rules across jurisdictions using multi-type vector similarity |
| **Graph Visualizer** | Interactive rule relationship graphs with Node2Vec embeddings |
| **Analytics Dashboard** | Rule clustering, coverage gaps, silhouette scores, and conflict resolution hints |

### Key Features

- **Executable Rules** — YAML-encoded decision trees with traceable legal citations
- **Multi-Jurisdiction** — EU (MiCA), UK (FCA), US (GENIUS/SEC/CFTC), Switzerland, Singapore, Hong Kong, Japan
- **Tiered Explanations** — Plain language for users, legal analysis for regulators
- **What-If Analysis** — Counterfactual engine simulates jurisdiction changes and threshold shifts
- **Semantic Search** — 4-type embeddings (semantic, structural, entity, legal) for rule discovery
- **Risk Assessment** — Market risk (VaR/CVaR), protocol risk, DeFi scoring, token compliance
- **Audit Trail** — Event-sourced versioning with temporal queries

## Architecture

```mermaid
%%{init: {
  "flowchart": {
    "nodeSpacing": 16,
    "rankSpacing": 22,
    "curve": "basis"
  }
}}%%

flowchart LR

subgraph Input[" "]
  direction TB
  DSL[YAML Rules]
  Corpus[Legal Corpus]
  DSL ~~~ Corpus
end

subgraph MLWorkflows["ML workflows"]
  direction LR

  subgraph EmbeddingDomain["embeddings/"]
    direction TB
    GEN[Generator] --> EMB4[4-Type Embeddings]
    EMB4 --> VS[Vector Search]
    EMB4 -.- SEM[semantic]
    EMB4 -.- STR[structural]
    EMB4 -.- ENT[entity]
    EMB4 -.- LEG[legal]
    GEN --> GEMB[Graph Embedding]
    GEMB --> N2V[Node2Vec]
    VS ~~~ N2V
  end

  subgraph VerifyDomain["verification/"]
    direction TB
    CE[Consistency Engine]
    CE --> T0[Tier 0: Schema]
    CE --> T1[Tier 1: Semantic]
    CE --> T2[Tier 2: Cross-rule]
    CE --> T3[Tier 3: Temporal]
    CE --> T4[Tier 4: External]
  end

  subgraph RAGDomain["rag/"]
    direction TB
    BM25[BM25 Index]
    BM25 --> CTX[Context Helpers]
  end

  subgraph DecoderDomain["decoder/"]
    direction TB
    DEC[Decoder] --> TMPL[Templates]
    DEC --> CIT[Citations]
    DEC --> CF[Counterfactual]
    CF --> DELTA[Delta]
    TMPL ~~~ CIT
    CIT ~~~ CF
  end
end

subgraph CoreEngine["Core rule execution"]
  direction LR

  subgraph CoreDomain["core/"]
    direction TB
    ONT[Ontology]
    TYPES[Types & Enums]
    RDSL[Rule DSL]
    TRC[Trace & Consistency]
    ONT --> TYPES --> RDSL
    RDSL ~~~ TRC
  end

  subgraph RulesDomain["rules/"]
    direction TB
    VER[Version Service]
    DE[Decision Engine]
    VER ~~~ DE
  end

  subgraph JurisdictionDomain["jurisdiction/"]
    direction TB
    JR[Resolver] --> JE[Evaluator]
    JE --> CD[Conflicts] --> PS[Pathway]
  end

  subgraph StorageDomain["storage/"]
    direction LR

    subgraph TemporalEngine["temporal/"]
      direction TB
      VREPO[Version Repo]
      EREPO[Event Repo]
      VREPO ~~~ EREPO
    end

    subgraph RetrievalEngine["retrieval/"]
      direction TB
      COMP[Compiler] --> IDX[Premise Index]
      COMP --> CACHE[IR Cache]
      IDX ~~~ CACHE
    end

    subgraph Stores["stores/"]
      direction TB
      JCFG[Jurisdiction Config]
      ESTORE[Embedding Store]
      GSTORE[Graph Store]
      JCFG ~~~ ESTORE
      ESTORE ~~~ GSTORE
    end
  end
end

subgraph RiskDomains["Risk assessment"]
  direction TB
  MR[market_risk/]
  PR[protocol_risk/]
  DR[defi_risk/]
  TC[token_compliance/]
  MR ~~~ PR
  PR ~~~ DR
  DR ~~~ TC
end

subgraph Output[" "]
  direction TB
  subgraph API["FastAPI"]
    direction TB
    R1["/decide"]
    R2["/rules"]
    R3["/verify"]
    R4["/search/*"]
    R5["/navigate"]
    R6["/embeddings"]
    R7["/graph"]
    R8["/decoder"]
    R9["/risk/*"]
  end
  ST[Streamlit UI]
  R1 -.-> ST
  R2 -.-> ST
  R3 -.-> ST
  R4 -.-> ST
  R5 -.-> ST
  R8 -.-> ST
  R9 -.-> ST
end

%% YAML rules derived from DSL layer
RDSL -.-> DSL

%% Main flows
DSL --> VER --> VREPO
VER --> EREPO
DSL --> COMP
DSL --> GEN

VREPO --> DE --> JE
JR <--> JCFG
PS <--> JCFG

Corpus --> CE
Corpus --> BM25

VS --> R4
VS <--> ESTORE
N2V <--> GSTORE
ESTORE --> R6
GSTORE --> R7

DE --> R1
PS --> R5
CE --> R3
CTX --> R2

%% Decoder connections
DE --> DEC
BM25 --> CIT
VS --> TMPL
DEC --> R8

%% Risk domains
MR --> R9
PR --> R9
DR --> R9
TC --> R9

%% Cross-subgraph packing
GEN ~~~ CE
CE ~~~ BM25
VER ~~~ COMP
DE ~~~ GEN
VREPO ~~~ COMP
ONT ~~~ GEN
RDSL ~~~ CE
BM25 ~~~ DEC
```

## Project Structure

```
backend/
├── main.py                      # FastAPI application entry point
├── config.py                    # Global settings
│
├── core/                        # Shared infrastructure
│   ├── config.py                # Settings & feature flags
│   ├── database.py              # SQLModel ORM (PostgreSQL/SQLite)
│   ├── models.py                # SQLModel entities
│   ├── ontology/                # Domain types (Actor, Instrument, Provision)
│   ├── visualization/           # Tree rendering (Graphviz, Mermaid)
│   └── api/                     # FastAPI routers
│       ├── routes_decide.py     # /decide endpoint
│       ├── routes_decoder.py    # /decoder endpoint
│       ├── routes_analytics.py  # /analytics endpoint
│       ├── routes_navigate.py   # /navigate cross-border navigation
│       ├── routes_risk.py       # /risk/* market risk endpoints
│       ├── routes_ke.py         # KE workbench endpoints
│       ├── routes_rules.py      # /rules endpoint
│       └── routes_production.py # /v2 production API
│
├── rules/                       # Rule management & evaluation
│   ├── service.py               # RuleLoader, DecisionEngine, TraceStep
│   ├── schemas.py               # Rule, ConditionSpec, ObligationSpec
│   ├── router.py                # Rules API router
│   ├── data/                    # YAML rule packs (MiCA, FCA, GENIUS, FINMA, MAS, RWA)
│   └── jurisdiction/            # Legacy jurisdiction (migrated to jurisdiction/)
│
├── jurisdiction/                # Cross-border compliance
│   └── service.py               # Resolver, Evaluator, Conflicts, Pathway
│
├── decoder/                     # ML-powered explanation layer
│   ├── service.py               # DecoderService, CounterfactualEngine
│   ├── schemas.py               # ExplanationTier, Citation, Templates
│   └── router.py                # Decoder API router
│
├── storage/                     # Persistence & compilation
│   ├── database.py              # SQLAlchemy connection (PostgreSQL/SQLite)
│   ├── migration.py             # YAML → database migration
│   ├── models.py                # Database models
│   ├── repositories/            # Data access layer
│   │   ├── rule_repo.py         # Rule CRUD operations
│   │   └── verification_repo.py # Verification CRUD
│   ├── retrieval/               # IR compilation & execution
│   │   ├── compiler/            # YAML → IR compilation
│   │   │   ├── compiler.py      # RuleCompiler
│   │   │   ├── ir.py            # Intermediate representation
│   │   │   ├── optimizer.py     # IR optimization
│   │   │   └── premise_index.py # O(1) premise lookup
│   │   └── runtime/             # IR cache & execution
│   │       ├── cache.py         # IR caching
│   │       ├── executor.py      # Rule execution
│   │       └── trace.py         # Execution tracing
│   ├── temporal/                # Event sourcing & versioning
│   │   ├── version_repo.py      # Rule version history
│   │   └── event_repo.py        # Audit trail events
│   └── stores/                  # Embedding & graph storage
│       ├── embedding_store.py   # Vector storage (5 types)
│       ├── graph_store.py       # Relationship graph
│       └── jurisdiction_config_repo.py  # Jurisdiction config
│
├── verification/                # Semantic consistency
│   └── service.py               # ConsistencyEngine (Tier 0-4 checks)
│
├── analytics/                   # Error patterns & drift
│   ├── service.py               # RuleAnalytics, Clustering
│   ├── error_patterns.py        # Pattern analysis
│   └── drift.py                 # Drift detection
│
├── rag/                         # Retrieval-augmented context
│   ├── service.py               # BM25Index, Retriever, AnswerGenerator
│   ├── corpus_loader.py         # Legal document loading
│   ├── rule_context.py          # Rule context retrieval
│   └── frontend_helpers.py      # UI helper functions
│
├── embeddings/                  # Vector embeddings & graph
│   ├── generator.py             # EmbeddingGenerator (4-type)
│   ├── service.py               # Embedding search service
│   └── graph.py                 # Node2Vec graph embeddings
│
├── market_risk/                 # Market risk analytics
│   ├── service.py               # VaR, CVaR, liquidity metrics
│   └── schemas.py               # Risk models
│
├── protocol_risk/               # Blockchain protocol assessment
│   ├── service.py               # Consensus, decentralization scoring
│   └── constants.py             # Protocol configurations
│
├── defi_risk/                   # DeFi protocol risk
│   ├── service.py               # Smart contract, oracle, governance scoring
│   └── constants.py             # Pre-configured protocols
│
├── token_compliance/            # Token standards compliance
│   └── (Howey Test, GENIUS Act analysis)
│
└── synthetic_data/              # Test data generation
    ├── base.py                  # Base generators
    ├── scenario_generator.py    # 500 test scenarios
    ├── rule_generator.py        # Synthetic rules
    └── verification_generator.py # Verification evidence

frontend/
├── Home.py                      # Landing page with framework coverage
├── pages/
│   ├── 1_KE_Workbench.py        # Rule verification & review
│   ├── 2_Production_Demo.py     # Synthetic data, guardrails, performance
│   ├── 3_Cross_Border_Navigator.py  # Multi-jurisdiction compliance
│   ├── 4_Embedding_Explorer.py  # UMAP visualization
│   ├── 5_Similarity_Search.py   # Multi-type similarity search
│   ├── 6_Graph_Visualizer.py    # Rule network graphs
│   └── 7_Analytics_Dashboard.py # Clustering & coverage analysis
├── ui/                          # Shared UI components
│   ├── embedding_viz.py         # UMAP rendering
│   ├── graph_components.py      # Graph visualization
│   └── similarity_cards.py      # Search result cards
└── helpers/
    └── analytics_client.py      # Analytics API client

data/legal/                      # Legal corpus (MiCA, FCA, GENIUS, FINMA, MAS)
docs/                            # Design documentation
tests/                           # Test suite (700+ tests)
```

## Regulatory Frameworks

| Framework | Jurisdiction | Rules | Accuracy | Status |
|-----------|--------------|-------|----------|--------|
| **MiCA** | EU | 8 | High | Enacted law (2023/1114) |
| **FCA Crypto** | UK | 5 | High | Enacted rules (COBS 4.12A) |
| **GENIUS Act** | US | 6 | High | Enacted law (July 2025) |
| **SEC Securities** | US_SEC | - | High | Securities Act 1933 |
| **CFTC Digital Assets** | US_CFTC | - | High | CFTC 2024 framework |
| **FINMA DLT** | CH | 6 | High | Enacted law (DLT Act 2021) |
| **MAS PSA** | SG | 6 | High | Enacted law (PSA 2019) |
| **SFC VASP** | HK | - | High | Hong Kong VASP regime 2023 |
| **PSA Japan** | JP | - | High | Payment Services Act 2023 |
| **RWA Tokenization** | EU | 3 | Low | Hypothetical framework |

### Synthetic Test Coverage

The `backend/synthetic_data/` package expands test coverage from ~30 to 500 scenarios:

| Category | Count | Description |
|----------|-------|-------------|
| happy_path | 120 | Valid compliant scenarios |
| edge_case | 120 | Threshold boundaries (€5M ART, 100% reserves) |
| negative | 80 | Rule violations |
| cross_border | 60 | Multi-jurisdiction conflicts |
| stablecoin | 30 | ART/EMT/algorithmic distinctions |
| exemption | 25 | Small offering, NFT, private placement |
| temporal | 25 | Version-dependent |
| defi | 20 | Decentralized protocol classification |
| passporting | 10 | EU CASP passporting |
| aml_travel_rule | 10 | KYC/originator data |

## Rule Embeddings

The system generates 4 types of vector embeddings per rule for multi-faceted similarity search:

| Type | Source | Use Case |
|------|--------|----------|
| **Semantic** | Name, description, decision explanation | Natural language search |
| **Structural** | Conditions, operators, decision logic | Find structurally similar rules |
| **Entity** | Field names, operators | Find rules using same data fields |
| **Legal** | Citations, document IDs | Find rules from same legal sources |

- Uses `sentence-transformers` (all-MiniLM-L6-v2) for dense embeddings
- Falls back to hash-based vectors when ML unavailable
- SQLite: JSON arrays; PostgreSQL: pgvector ready

## Backend Domains

| Domain | Purpose | Key Components |
|--------|---------|----------------|
| **core/** | Shared infrastructure | Config, SQLModel ORM, ontology types, FastAPI routes |
| **rules/** | Rule evaluation | `RuleLoader`, `DecisionEngine`, YAML parser, `TraceStep` |
| **jurisdiction/** | Cross-border compliance | Resolver, Evaluator, Conflicts, Pathway synthesis |
| **decoder/** | ML explanations | `DecoderService`, `CounterfactualEngine`, citations, templates |
| **storage/** | Persistence & compilation | SQLAlchemy, repositories, IR compiler, temporal engine, stores |
| **verification/** | Consistency | `ConsistencyEngine` (Tier 0-4 semantic validation) |
| **analytics/** | Analysis | Rule clustering, drift detection, error patterns |
| **rag/** | Retrieval | `BM25Index`, `Retriever`, context helpers, corpus loader |
| **embeddings/** | Embeddings | `EmbeddingGenerator` (4-type vectors), Node2Vec graph |
| **market_risk/** | Market analytics | VaR, CVaR, liquidity metrics, correlations |
| **protocol_risk/** | Blockchain assessment | Consensus scoring, decentralization, settlement finality |
| **defi_risk/** | DeFi protocol risk | Smart contract, oracle, governance scoring (A-F grades) |
| **token_compliance/** | Token standards | Howey Test, GENIUS Act stablecoin analysis |
| **synthetic_data/** | Test generation | Scenario/rule generators, verification evidence |

### Database Support

The backend supports both SQLite (local development) and PostgreSQL (production):

```bash
# Local development (default)
DATABASE_URL=sqlite:///./data/ke_workbench.db

# Production (Railway auto-injects)
DATABASE_URL=postgresql://user:pass@host:5432/db
```

## Quick Start

### Local Development

```bash
# Clone and install
git clone https://github.com/yourusername/droit.git
cd droit
pip install -r requirements.txt

# Start API server
uvicorn backend.main:app --reload

# Start Streamlit frontend (separate terminal)
streamlit run frontend/Home.py
```

### Railway Deployment

The project is configured for [Railway](https://railway.app) deployment:

- **Dockerfile**: Multi-stage build with Python 3.11
- **railway.json**: Health checks at `/health`, auto-restart on failure
- **PostgreSQL**: Automatically provisioned via Railway plugin

```bash
# Deploy to Railway
railway up

# Or connect existing repo via Railway dashboard
```

The live demo at [pazooki.streamlit.app](https://pazooki.streamlit.app) runs on Railway with PostgreSQL.

## Documentation

- [Knowledge Model](docs/knowledge_model.md) — Ontology design
- [Rule DSL](docs/rule_dsl.md) — YAML rule specification
- [Engine Design](docs/engine_design.md) — Architecture details
- [Embedding Service](docs/embedding_service.md) — Vector search design
- [Market Risk Modules](docs/market_risk_modules.md) — VaR, protocol risk, DeFi scoring
- [Synthetic Data Strategy](docs/SYNTHETIC_DATA_STRATEGY.md) — Test coverage expansion
- [Migration Plan](docs/MIGRATION_PLAN_FLAT_DOMAINS.md) — Flat domain architecture migration

## Disclaimer

Research/demo project, not legal advice. Rules are interpretive models—consult qualified legal counsel for compliance decisions.

## License

MIT License. See [LICENSE](LICENSE).

---

Built with [Claude Code](https://claude.ai/code)

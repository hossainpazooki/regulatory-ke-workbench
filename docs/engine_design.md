# Engine Design: Internal Knowledge Engineering Workbench

This document describes the architecture of the internal Knowledge Engineering (KE) workbench for regulatory reasoning.

## System Overview

The workbench provides tools for Knowledge Engineers to:
- Create and maintain regulatory rules
- Verify rule consistency against source legal text
- Monitor rule quality over time
- Prioritize rules for human review

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 6: KE Interfaces                       │
│      (FastAPI /ke endpoints, Streamlit dashboard, CLI tools)    │
├─────────────────────────────────────────────────────────────────┤
│                 Layer 5: Visualization (Optional)               │
│      (Supertree charts, tree adapters, HTML rendering)          │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 4: Internal RAG                        │
│           (RuleContextRetriever, document indexing)             │
├─────────────────────────────────────────────────────────────────┤
│              Layer 3B: Semantic Consistency Engine              │
│     (ConsistencyEngine, Tier 0-4 verification, analytics)       │
├─────────────────────────────────────────────────────────────────┤
│              Layer 3A: Symbolic Decision Engine                 │
│          (DecisionEngine, RuleLoader, condition eval)           │
├─────────────────────────────────────────────────────────────────┤
│                   Production Layer                              │
│   (Persistence, Compiler, Runtime, Premise Index, IR Cache)     │
├─────────────────────────────────────────────────────────────────┤
│                Layer 2: Rule DSL (YAML + Pydantic)              │
│           (Rule, ConditionSpec, DecisionTree, etc.)             │
├─────────────────────────────────────────────────────────────────┤
│                  Layer 1: Ontology (OCaml + Python)             │
│        (Actor, Instrument, Activity, Provision, etc.)           │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 3A: Symbolic Decision Engine

The decision engine evaluates rules against scenarios deterministically.

### Key Components

- **RuleLoader** ([loader.py](../backend/rules/loader.py)): Loads YAML rules, parses decision trees and conditions
- **DecisionEngine** ([engine.py](../backend/rules/engine.py)): Evaluates scenarios against rules
- **Rule Schema** ([schema.py](../backend/rules/schema.py)): Pydantic models for rule structure

### Decision Flow

```
Scenario → RuleLoader → Applicable Rules → Decision Tree Traversal → DecisionResult
```

### Output Structure

```python
class DecisionResult:
    rule_id: str
    applicable: bool
    decision: str | None
    trace: list[TraceStep]       # Decision path for explainability
    obligations: list[Obligation]
    rule_metadata: RuleMetadata  # Includes consistency status
```

## Production Layer: Persistence, Compiler, and Runtime

The production layer provides compile-time/runtime separation for efficient rule evaluation with O(1) lookup and linear condition evaluation.

### Components

| Component | Module | Purpose |
|-----------|--------|---------|
| **Database** | `backend/persistence/database.py` | SQLite (PostgreSQL-compatible) connection |
| **Rule Repository** | `backend/persistence/repositories/rule_repo.py` | Rule CRUD, premise index storage |
| **Verification Repository** | `backend/persistence/repositories/verification_repo.py` | Verification results, human reviews |
| **Compiler** | `backend/compiler/compiler.py` | AST → IR compilation |
| **IR Types** | `backend/compiler/ir.py` | Intermediate representation models |
| **Premise Index** | `backend/compiler/premise_index.py` | Inverted index for O(1) lookup |
| **Optimizer** | `backend/compiler/optimizer.py` | Condition flattening, selectivity ordering |
| **Runtime** | `backend/runtime/executor.py` | Linear IR evaluation |
| **IR Cache** | `backend/runtime/cache.py` | Thread-safe in-memory caching |
| **Trace** | `backend/runtime/trace.py` | Execution tracing for auditability |

### Database Schema

```sql
-- Core rule storage
CREATE TABLE rules (
    id TEXT PRIMARY KEY,
    rule_id TEXT UNIQUE NOT NULL,
    version INTEGER DEFAULT 1,
    content_yaml TEXT NOT NULL,
    content_json TEXT,
    rule_ir TEXT,                   -- Compiled IR (JSON)
    compiled_at TEXT,
    source_document_id TEXT,
    source_article TEXT,
    is_active INTEGER DEFAULT 1
);

-- Premise index for O(1) lookup
CREATE TABLE rule_premise_index (
    premise_key TEXT NOT NULL,      -- e.g., "instrument_type:art"
    rule_id TEXT NOT NULL,
    PRIMARY KEY (premise_key, rule_id)
);

-- Verification results
CREATE TABLE verification_results (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL,
    verified_at TEXT,
    verified_by TEXT
);

-- Evidence records
CREATE TABLE verification_evidence (
    id TEXT PRIMARY KEY,
    verification_id TEXT NOT NULL,
    tier INTEGER NOT NULL,
    category TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL,
    details TEXT
);

-- Human reviews
CREATE TABLE reviews (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    notes TEXT,
    created_at TEXT
);
```

### Intermediate Representation (IR)

Rules are compiled to IR for efficient runtime execution:

```python
class RuleIR(BaseModel):
    rule_id: str
    version: int
    ir_version: int = 1

    # O(1) lookup keys
    premise_keys: list[str]  # ["instrument_type:art", "activity:public_offer"]

    # Flattened applicability checks
    applicability_checks: list[CompiledCheck]
    applicability_mode: Literal['all', 'any']

    # Decision table (replaces tree traversal)
    decision_table: list[DecisionEntry]

    compiled_at: str
    source_hash: str
```

### Premise Index

The premise index enables O(1) rule candidate lookup:

```python
# Given facts: {"instrument_type": "art", "activity": "public_offer"}
# Build premise keys: ["instrument_type:art", "activity:public_offer"]
# Lookup: O(1) set intersection → candidate rule IDs
```

### Compilation Flow

```
YAML Rule → Parse (RuleLoader) → Rule AST → Compile (RuleCompiler) → RuleIR
                                                ↓
                              Store IR + Premise Keys in Database
```

### Runtime Execution Flow

```
Facts → Premise Index Lookup (O(1)) → Candidate Rules
                ↓
    Load IR from Cache/Database
                ↓
    Linear Applicability Check → Decision Table Match
                ↓
           DecisionResult + Trace
```

### API Endpoints (v2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/migrate` | POST | Migrate YAML rules to database |
| `/v2/status` | GET | Database statistics |
| `/v2/rules/{id}/compile` | POST | Compile single rule to IR |
| `/v2/rules/compile` | POST | Compile all rules |
| `/v2/rules/{id}/evaluate` | POST | Evaluate with compiled IR |
| `/v2/evaluate` | POST | Batch evaluation with premise index |
| `/v2/cache/stats` | GET | IR cache statistics |
| `/v2/cache/clear` | POST | Clear IR cache |
| `/v2/cache/preload` | POST | Preload all IR to cache |
| `/v2/index/stats` | GET | Premise index statistics |
| `/v2/index/rebuild` | POST | Rebuild premise index |
| `/v2/index/lookup` | GET | Look up rules by premise |

### Performance Characteristics

| Operation | Original Engine | Production Runtime |
|-----------|----------------|-------------------|
| Rule lookup | O(n) scan | O(1) premise index |
| Condition eval | Recursive tree | Linear check array |
| 'in' operator | O(n) list scan | O(1) set lookup |
| Rule load | YAML parse | IR deserialization |

## Layer 3B: Semantic Consistency Engine

Verifies that rules accurately represent source legal text.

### Components

- **ConsistencyEngine** ([consistency_engine.py](../backend/verify/consistency_engine.py)): Runs verification checks
- **ErrorPatternAnalyzer** ([error_patterns.py](../backend/analytics/error_patterns.py)): Detects systematic issues
- **DriftDetector** ([drift.py](../backend/analytics/drift.py)): Tracks quality changes over time

### Verification Tiers

| Tier | Status | Description |
|------|--------|-------------|
| 0 | Implemented | Schema validation, required fields, date consistency |
| 1 | Implemented | Deontic alignment, keyword overlap, negation consistency |
| 2 | Stub | Semantic similarity (requires sentence-transformers) |
| 3 | Stub | NLI entailment (requires NLI model) |
| 4 | Stub | Cross-rule consistency |

### Consistency Block Structure

Every rule can have a `consistency` block:

```yaml
consistency:
  summary:
    status: verified | needs_review | inconsistent | unverified
    confidence: 0.95
    last_verified: "2024-12-10T14:30:00Z"
    verified_by: system | human:username
  evidence:
    - tier: 0
      category: schema_valid
      label: pass
      score: 1.0
      details: "All required fields present"
```

### Key Design Decisions

1. **No external LLM calls**: All verification is local
2. **Deterministic base**: Tier 0-1 are rule-based, not ML-based
3. **Writeback**: Results are stored in rule YAML files
4. **Human override**: Humans can mark rules as verified regardless of automated checks

## Layer 4: Internal RAG

Provides context retrieval for rule verification - NOT for public Q&A.

### Components

- **RuleContextRetriever** ([rule_context.py](../backend/rag/rule_context.py)): Rule-specific retrieval
- **BM25Index** ([bm25.py](../backend/rag/bm25.py)): Keyword-based retrieval
- **Retriever** ([retriever.py](../backend/rag/retriever.py)): Hybrid BM25 + optional vectors

### Usage Pattern

```python
retriever = RuleContextRetriever(rule_loader=loader)
retriever.index_document("mica_2023", mica_text)

# Get source context for a rule
context = retriever.get_rule_context(rule)
source_text = retriever.get_source_text(rule)

# Pass to consistency engine
result = consistency_engine.verify_rule(rule, source_text)
```

### Capabilities

- Index legal documents by article/section
- Retrieve source passages for rules
- Find cross-references in text
- Locate related rules by source/tags

### KE Workbench UI Integration

The internal RAG layer powers several KE workbench UI features:

- **Source & Context panel**: Displays the primary text span backing a rule, with before/after context paragraphs and document/article metadata.
- **Similar / related provisions panel**: Uses structural filtering (same document_id) and similarity thresholds to show related rules without noise. Displays "no results above threshold" when appropriate.
- **Corpus search (sidebar)**: Supports dual-mode search:
  - *Article lookup mode*: Queries like "Art. 36(1)" or "Article 45" perform exact article matching against rule `source.article` fields.
  - *Semantic search mode*: Natural language queries perform BM25 retrieval, with results mapped back to rules via `(document_id, article)` matching.

**Important**: Internal RAG is NOT exposed as a public `/ask` endpoint in this repo. It is strictly for KE tooling.

### Legal Corpus Integration

The workbench includes a small embedded legal corpus for MiCA, the EU DLT Pilot Regime, and the GENIUS Act (US stablecoin framework). These are normalized excerpts, not full official texts.

#### Corpus Structure

```
data/legal/
├── mica_2023/
│   ├── meta.yaml           # Document metadata
│   └── text_normalized.txt # Normalized excerpts
├── dlt_pilot_2022/
│   ├── meta.yaml
│   └── text_normalized.txt
└── genius_act_2025/
    ├── meta.yaml
    └── text_normalized.txt
```

Each `meta.yaml` contains:
- `document_id`: Join key to rule `source.document_id`
- `title`: Human-readable document title
- `citation`: Official citation (e.g., "Regulation (EU) 2023/1114")
- `jurisdiction`: "EU" or "US"
- `source_url`: Link to official text

#### Corpus Loader

```python
from backend.rag_service.app.services import load_legal_document, load_all_legal_documents

# Load a specific document
doc = load_legal_document("mica_2023")
print(doc.title, doc.citation)
print(doc.find_article_text("36"))  # Get Article 36 text

# Load all documents
docs = load_all_legal_documents()
```

#### Rule-Corpus Mapping

Rules reference legal corpus via `source.document_id`:
- MiCA rules: `document_id: mica_2023`
- DLT Pilot rules: `document_id: dlt_pilot_2022`
- GENIUS rules: `document_id: genius_act_2025`

The `RuleLoader.validate_corpus_coverage()` method checks which rules have corresponding legal corpus entries.

#### Coverage Gap Detection

When searching the corpus, hits are tagged with:
- `source_type: "legal_text"` for legal corpus hits
- `has_rule_coverage: False` when a legal passage has no mapped rule

This enables gap-finding UX: show ⚠️ "no formal rule yet" for legal text passages without corresponding rules.

## Layer 5: Visualization (Optional)

Provides tree-based visualizations for regulatory charts. Gracefully degrades when Supertree is not installed.

### Components

- **supertree_adapters.py**: Pure-Python adapters that convert rules/traces into nested dict/list structures
- **supertree_utils.py**: Rendering helpers with optional Supertree dependency

### Available Charts

| Chart | Function | Description |
|-------|----------|-------------|
| Rulebook Outline | `build_rulebook_outline()` | Hierarchical view of rules by document |
| Decision Trace | `build_decision_trace_tree()` | Evaluation path through a rule |
| Ontology Browser | `build_ontology_tree()` | Actor/Instrument/Activity type hierarchy |
| Corpus Links | `build_corpus_rule_links()` | Document → Article → Rule traceability |
| Decision Tree | `build_decision_tree_structure()` | Rule's internal decision logic |

### Optional Dependency

```bash
pip install -r requirements-visualization.txt
```

When Supertree is not installed:
- `is_supertree_available()` returns `False`
- Render functions return fallback HTML with install instructions
- Data adapters work normally (no external dependencies)

### Usage Pattern

```python
from backend.visualization import (
    build_rulebook_outline,
    render_rulebook_outline_html,
    is_supertree_available,
)

# Get tree data (always works)
rules = loader.get_all_rules()
tree_data = build_rulebook_outline(rules)

# Render HTML (graceful fallback if Supertree not installed)
html = render_rulebook_outline_html(tree_data)
```

## Layer 6: KE Interfaces

### API Endpoints

The `/ke` prefix provides internal endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ke/verify` | POST | Verify single rule |
| `/ke/verify-all` | POST | Verify all rules |
| `/ke/analytics/summary` | GET | Overall statistics |
| `/ke/analytics/patterns` | GET | Detected error patterns |
| `/ke/analytics/matrix` | GET | Category × outcome matrix |
| `/ke/analytics/review-queue` | GET | Prioritized review list |
| `/ke/drift/baseline` | POST | Set drift baseline |
| `/ke/drift/detect` | GET | Detect drift from baseline |
| `/ke/drift/history` | GET | Metrics history |
| `/ke/drift/authors` | GET | Per-author statistics |
| `/ke/context/{rule_id}` | GET | Rule source context |
| `/ke/related/{rule_id}` | GET | Related rules |
| `/ke/rules/{rule_id}/review` | POST | Submit human review |
| `/ke/rules/{rule_id}/reviews` | GET | Get review history |
| `/ke/charts/supertree-status` | GET | Check Supertree availability |
| `/ke/charts/rulebook-outline` | GET | Rulebook outline tree data |
| `/ke/charts/rulebook-outline/html` | GET | Rulebook outline as HTML |
| `/ke/charts/ontology` | GET | Ontology tree data |
| `/ke/charts/ontology/html` | GET | Ontology as HTML |
| `/ke/charts/corpus-links` | GET | Corpus-rule links tree data |
| `/ke/charts/corpus-links/html` | GET | Corpus-rule links as HTML |
| `/ke/charts/decision-tree/{rule_id}` | GET | Decision tree for a rule |
| `/ke/charts/decision-trace/{rule_id}` | POST | Evaluate and get trace tree |
| `/ke/charts/decision-trace/{rule_id}/html` | POST | Trace as HTML |

### Review Queue Priority

Rules are prioritized for review based on:
1. Consistency status (inconsistent > needs_review > unverified)
2. Confidence score (lower = higher priority)
3. Time since last verification
4. Rule importance (future: usage frequency)

## Analytics

### Error Pattern Detection

Identifies systematic issues across rules:

```python
analyzer = ErrorPatternAnalyzer(rule_loader=loader)
patterns = analyzer.detect_patterns(min_affected=2)

# Example pattern:
# {
#   "pattern_id": "high_fail_deontic_alignment",
#   "category": "deontic_alignment",
#   "severity": "high",
#   "affected_rule_count": 5,
#   "recommendation": "Review deontic verb usage vs rule modality"
# }
```

### Drift Detection

Tracks quality changes over time:

```python
detector = DriftDetector(rule_loader=loader)
detector.set_baseline()  # Capture initial state

# Later...
report = detector.detect_drift()
# report.drift_severity: "none" | "minor" | "moderate" | "major"
```

## Testing Strategy

The workbench has comprehensive test coverage:

| Test File | Coverage |
|-----------|----------|
| `test_rules_schema.py` | Consistency models, save/load |
| `test_consistency_engine.py` | Tier 0-1 checks, summary computation |
| `test_rag_internal.py` | Context retrieval, cross-references |
| `test_analytics.py` | Error patterns, drift detection |
| `test_api_ke.py` | All KE endpoints |
| `test_persistence.py` | Database CRUD, migration, premise index |
| `test_compiler.py` | IR generation, premise extraction, optimizer |
| `test_runtime.py` | IR execution, cache, trace generation |

Run all tests:
```bash
pytest tests/ -v
```

## Future Enhancements

### Tier 2+: ML-Based Verification

When ML dependencies are available:
- Semantic similarity via sentence-transformers
- NLI-based entailment checking
- Cross-rule contradiction detection

### Confident Learning Integration

Per the spec in `semantic_consistency_regulatory_kg.md`:
- Identify likely label errors in rule annotations
- Estimate per-category noise rates
- Generate cleaned training data for ML models

### CLI Tools

Future CLI commands for KE workflows:
- `droit verify <rule_id>` - Verify single rule
- `droit verify --all` - Verify all rules
- `droit review` - Interactive review queue
- `droit drift` - Show drift report

## Integration with OCaml Core

The Python workbench reads from YAML rules that are the executable form of the OCaml type system:

```
OCaml ontology.ml  ───►  docs/ontology_design.md
        │
        ▼
OCaml rule_dsl.ml  ───►  YAML rule files  ◄───  Python loader
        │                      │
        ▼                      ▼
  docs/rule_dsl.md      Python DecisionEngine
```

The OCaml types remain the source of truth. Python mirrors them via Pydantic models but defers to OCaml for formal verification (future).

## Multi-Jurisdiction Navigation (v4 Architecture)

The v4 architecture extends the system with cross-border compliance navigation capabilities.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    /navigate Endpoint                           │
│              (Cross-border compliance navigation)                │
├─────────────────────────────────────────────────────────────────┤
│                Jurisdiction Module                              │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│   │  Resolver   │  │  Evaluator  │  │ Conflict Detector   │    │
│   │  (resolve   │  │  (parallel  │  │ (classification,    │    │
│   │   jurisd.)  │  │   eval)     │  │  obligation, etc.)  │    │
│   └─────────────┘  └─────────────┘  └─────────────────────┘    │
│                          │                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Pathway Synthesizer                         │  │
│   │   (ordered steps, timeline, critical path, waivers)      │  │
│   └─────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Premise Index                                │
│      (O(1) lookup with jurisdiction and regime filtering)       │
├─────────────────────────────────────────────────────────────────┤
│                    Jurisdiction Registry                        │
│      (SQLite tables: jurisdictions, regimes, equivalences)      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### Jurisdiction Resolver (`backend/jurisdiction/resolver.py`)

Resolves which jurisdictions and regulatory regimes apply:

```python
from backend.jurisdiction.resolver import resolve_jurisdictions

applicable = resolve_jurisdictions(
    issuer="CH",           # Swiss issuer
    targets=["EU", "UK"],  # Targeting EU and UK markets
    instrument_type="stablecoin",
)
# Returns: CH (issuer_home), EU (target), UK (target)
```

#### Jurisdiction Evaluator (`backend/jurisdiction/evaluator.py`)

Parallel evaluation across jurisdictions with O(1) premise index lookup:

```python
from backend.jurisdiction.evaluator import evaluate_jurisdiction

result = await evaluate_jurisdiction(
    jurisdiction="UK",
    regime_id="fca_crypto_2024",
    facts={"instrument_type": "crypto_asset", ...},
)
```

#### Conflict Detector (`backend/jurisdiction/conflicts.py`)

Detects cross-jurisdiction conflicts:

- **Classification divergence**: Same instrument, different regulatory treatment
- **Obligation conflicts**: Incompatible requirements
- **Timeline conflicts**: Conflicting deadlines
- **Decision conflicts**: Permitted in one jurisdiction, prohibited in another

#### Pathway Synthesizer (`backend/jurisdiction/pathway.py`)

Generates ordered compliance roadmap:

```python
from backend.jurisdiction.pathway import synthesize_pathway

pathway = synthesize_pathway(
    results=jurisdiction_results,
    conflicts=conflicts,
    equivalences=equivalences,
)
# Returns: ordered steps with dependencies, timelines, waivers
```

### API Endpoint

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/navigate` | POST | Cross-border compliance navigation |
| `/navigate/jurisdictions` | GET | List supported jurisdictions |
| `/navigate/regimes` | GET | List regulatory regimes |
| `/navigate/equivalences` | GET | List equivalence determinations |

### Database Schema (v4)

New tables for multi-jurisdiction support:

- `jurisdictions`: Jurisdiction registry (EU, UK, US, CH, SG)
- `regulatory_regimes`: Regime metadata (mica_2023, fca_crypto_2024, etc.)
- `equivalence_determinations`: Cross-border equivalence decisions
- `rule_conflicts`: Known rule conflicts

### Premise Index with Jurisdiction

The premise index now includes jurisdiction keys for O(1) filtered lookup:

```python
# Premise keys include jurisdiction and regime
premise_keys = [
    "jurisdiction:EU",
    "regime:mica_2023",
    "instrument_type:stablecoin",
]

# Lookup by jurisdiction
rules = premise_index.lookup_by_jurisdiction(facts, jurisdiction="UK")
```

### Streamlit Navigator

The Navigator page (`frontend/pages/3_Navigator.py`) provides a UI for:

1. Defining cross-border scenarios (issuer, targets, instrument type)
2. Running multi-jurisdiction analysis
3. Viewing jurisdiction-specific results
4. Reviewing detected conflicts
5. Visualizing compliance pathway
6. Estimating timeline to compliance

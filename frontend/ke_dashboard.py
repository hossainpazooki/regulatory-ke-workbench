"""
Home - Regulatory Knowledge Engineering Workbench.

Landing page with overview, instructions, and quick navigation.

Run from repo root:
    streamlit run frontend/ke_dashboard.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="KE Workbench",
    page_icon="🏠",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title("Regulatory Knowledge Engineering Workbench")

st.markdown("""
**A Computational Law Platform for MiCA, RWA Tokenization, and Stablecoin Frameworks**

Transform regulatory documents into executable knowledge through ontology extraction,
declarative rules, and traceable decision logic.
""")

st.divider()

# -----------------------------------------------------------------------------
# Quick Navigation
# -----------------------------------------------------------------------------

st.header("Quick Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    ### KE Workbench

    The main workbench for knowledge engineers to:
    - Browse and select rules by document
    - Visualize decision trees
    - Run trace tests on scenarios
    - Review verification evidence
    - Submit human reviews

    **Best for:** Day-to-day rule inspection and verification
    """)
    st.page_link("pages/1_KE_Workbench.py", label="Open KE Workbench", icon="⚖️")

with nav_col2:
    st.markdown("""
    ### Production Demo

    Explore the production architecture:
    - Compile rules to IR
    - View premise index statistics
    - Monitor cache hit rates
    - Benchmark performance
    - Compare O(1) vs O(n) lookup

    **Best for:** Understanding system scalability
    """)
    st.page_link("pages/2_Production_Demo.py", label="Open Production Demo", icon="🏭")

with nav_col3:
    st.markdown("""
    ### Charts & Analytics

    Visual analysis tools:
    - Rulebook outline with coverage
    - Ontology type browser
    - Corpus-to-rule links
    - Legal corpus coverage gaps

    **Best for:** High-level regulatory coverage analysis
    """)

st.divider()

# -----------------------------------------------------------------------------
# System Overview
# -----------------------------------------------------------------------------

st.header("System Overview")

overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    st.subheader("What This System Does")
    st.markdown("""
    Financial regulation is complex, multi-jurisdictional, and constantly evolving.
    This system encodes regulations as **executable rules** with full traceability
    back to source legal text.

    **Key Capabilities:**

    | Capability | Description |
    |------------|-------------|
    | **Automated Compliance** | Check scenarios against real regulatory frameworks |
    | **Decision Tracing** | Every decision shows which provisions applied and why |
    | **Semantic Verification** | Ensure rules faithfully represent source provisions |
    | **Gap Analysis** | Identify legal provisions without rule coverage |
    """)

with overview_col2:
    st.subheader("Architecture Diagram")
    st.code("""
    ┌─────────────────┐     ┌─────────────────┐
    │  Legal Corpus   │     │   YAML Rules    │
    │  (MiCA, GENIUS) │     │  (Decision DSL) │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  RAG Retrieval  │────▶│ Consistency     │
    │  (Context)      │     │ Engine          │
    └─────────────────┘     └────────┬────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Decision       │     │  Compiler       │     │  Premise Index  │
    │  Engine         │     │  (YAML → IR)    │     │  (O(1) Lookup)  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
    """, language=None)

st.divider()

# -----------------------------------------------------------------------------
# Regulatory Frameworks
# -----------------------------------------------------------------------------

st.header("Regulatory Frameworks Modeled")

framework_data = [
    {
        "Framework": "MiCA (EU)",
        "Document ID": "mica_2023",
        "Rules": "9 rules",
        "Coverage": "Public offers, ARTs, EMTs, CASPs, Market abuse",
        "Status": "Modeled",
    },
    {
        "Framework": "GENIUS Act (US)",
        "Document ID": "genius_act_2025",
        "Rules": "6 rules",
        "Coverage": "Stablecoin issuers, reserves, redemption, AML",
        "Status": "Illustrative",
    },
    {
        "Framework": "RWA Tokenization",
        "Document ID": "rwa_eu_2025",
        "Rules": "2 rules",
        "Coverage": "Authorization, custody requirements",
        "Status": "Illustrative",
    },
    {
        "Framework": "DLT Pilot",
        "Document ID": "dlt_pilot_2022",
        "Rules": "—",
        "Coverage": "Corpus only (future modeling)",
        "Status": "Planned",
    },
]

import pandas as pd
st.dataframe(pd.DataFrame(framework_data), use_container_width=True, hide_index=True)

st.divider()

# -----------------------------------------------------------------------------
# How to Use
# -----------------------------------------------------------------------------

st.header("How to Use")

tab_workflow, tab_verify, tab_test = st.tabs([
    "Basic Workflow",
    "Verify Rules",
    "Test Scenarios"
])

with tab_workflow:
    st.markdown("""
    ### Basic Workflow

    ```
    1. Navigate to KE Workbench
           │
           ▼
    2. Select a rule from the left panel
       ├── Queue view: Prioritized by verification status
       └── Navigator view: Browse by document hierarchy
           │
           ▼
    3. Review the decision tree
       └── Toggle "Overlay" to see consistency status
           │
           ▼
    4. Run trace tests (optional)
       └── Enter scenario values → Run Evaluation
           │
           ▼
    5. Check Analytics tab
       └── View evidence, confidence scores
           │
           ▼
    6. Submit review decision
       └── Mark as verified or flag issues
    ```
    """)

with tab_verify:
    st.markdown("""
    ### Verification Process

    The system performs **tiered semantic consistency checks**:

    | Tier | Type | Checks |
    |------|------|--------|
    | **0** | Schema | Valid YAML, required fields, date consistency |
    | **1** | Lexical | Deontic alignment, keyword overlap, negation |
    | **2** | Semantic | Embedding similarity (requires ML deps) |
    | **3** | NLI | Entailment checking (stub) |
    | **4** | Cross-rule | Inter-rule consistency (stub) |

    **To verify rules:**
    1. Go to KE Workbench
    2. Click **Verify All** in the header (or **Verify** on individual rule)
    3. Review evidence in the Analytics tab
    4. Status indicators: `?` needs review, `✓` verified, `✗` inconsistent
    """)

with tab_test:
    st.markdown("""
    ### Testing Scenarios

    The Trace/Test tab lets you evaluate rules against custom scenarios:

    **Step 1: Configure scenario**
    ```
    Instrument Type: [art, emt, stablecoin, payment_stablecoin, ...]
    Jurisdiction:    [EU, US, UK]
    Activity:        [public_offer, issuance, redemption, custody, ...]
    Entity Type:     [issuer, casp, investor]
    ```

    **Step 2: Set additional attributes**
    - Credit Institution: Whether the entity is a licensed bank
    - Authorized: Whether already authorized
    - Custodian Authorized: Whether custodian is authorized

    **Step 3: Run Evaluation**
    - View step-by-step trace through decision tree
    - See which conditions passed/failed
    - Check resulting decision and obligations
    """)

st.divider()

# -----------------------------------------------------------------------------
# Production Architecture
# -----------------------------------------------------------------------------

st.header("Production Architecture")

st.markdown("""
The system includes production-grade features for high-throughput compliance checking:
""")

prod_col1, prod_col2, prod_col3 = st.columns(3)

with prod_col1:
    st.markdown("""
    ### Premise Index

    **O(1) rule lookup** via inverted index

    ```
    fact pattern → matching rules

    "instrument_type:art" → [rule1, rule2]
    "jurisdiction:EU"     → [rule1, rule3]
    ```

    Instead of scanning all rules, lookup
    candidates in constant time.
    """)

with prod_col2:
    st.markdown("""
    ### Compiled IR

    **Linear condition evaluation**

    ```
    YAML Rule → Compiled IR

    - Flattened checks (no recursion)
    - Pre-computed value sets
    - Decision table lookup
    ```

    No tree traversal at runtime.
    """)

with prod_col3:
    st.markdown("""
    ### IR Cache

    **In-memory rule storage**

    ```
    Cache Stats:
    - Size: 17 rules
    - Hit rate: 98%
    - Misses: 2
    ```

    Eliminates parsing and DB lookups.
    """)

st.info("**Try it:** Go to **Production Demo** to compile rules and see performance benchmarks.")

st.divider()

# -----------------------------------------------------------------------------
# Rule DSL Example
# -----------------------------------------------------------------------------

st.header("Rule DSL Example")

st.markdown("Rules are defined in YAML with a declarative decision tree syntax:")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.markdown("**Rule Definition (YAML)**")
    st.code("""
- rule_id: mica_art36_public_offer_auth
  version: "1.0"
  description: |
    Authorization requirements for public offers
    of crypto-assets under MiCA Article 36.
  effective_from: 2024-06-30
  tags: [mica, authorization, public_offer]

  applies_if:
    all:
      - field: instrument_type
        operator: in
        value: [art, emt, crypto_asset]
      - field: activity
        operator: "=="
        value: public_offer
      - field: jurisdiction
        operator: "=="
        value: EU

  decision_tree:
    node_id: check_exemption
    condition:
      field: is_credit_institution
      operator: "=="
      value: true
    true_branch:
      result: exempt
      notes: "Credit institutions exempt"
    false_branch:
      node_id: check_authorization
      condition:
        field: authorized
        operator: "=="
        value: true
      true_branch:
        result: permitted
      false_branch:
        result: not_permitted
        obligations:
          - id: obtain_authorization
            description: "Apply for authorization"
    """, language="yaml")

with example_col2:
    st.markdown("**Decision Tree Visualization**")
    st.code("""
            ┌─────────────────────┐
            │   check_exemption   │
            │ is_credit_inst==T   │
            └──────────┬──────────┘
                 ┌─────┴─────┐
              TRUE         FALSE
                 │           │
            ┌────┴────┐ ┌────┴────────────┐
            │ exempt  │ │ check_auth      │
            │         │ │ authorized==T   │
            └─────────┘ └────────┬────────┘
                            ┌────┴────┐
                         TRUE      FALSE
                            │         │
                       ┌────┴────┐ ┌──┴──────────┐
                       │permitted│ │not_permitted│
                       └─────────┘ │+ obligation │
                                   └─────────────┘
    """, language=None)

    st.markdown("**Trace Output**")
    st.code("""
    Scenario: {instrument_type: art,
               activity: public_offer,
               jurisdiction: EU,
               is_credit_institution: false,
               authorized: false}

    Trace:
    ✓ instrument_type in [art, emt, crypto_asset]
    ✓ activity == public_offer
    ✓ jurisdiction == EU
    ✗ is_credit_institution == true → FALSE
    ✗ authorized == true → FALSE

    Decision: not_permitted
    Obligations: [obtain_authorization]
    """, language=None)

st.divider()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("""
---

### Resources

- **Documentation:** See `docs/` folder for detailed specs
- **Rule DSL:** [docs/rule_dsl.md](docs/rule_dsl.md)
- **Knowledge Model:** [docs/knowledge_model.md](docs/knowledge_model.md)
- **API Endpoints:** Run `uvicorn backend.main:app --reload` and visit `/docs`

### Disclaimers

This is a research/demo project, not legal advice. Rules are interpretive models
of regulatory text, not authoritative legal guidance. Always consult qualified
legal counsel for compliance decisions.

---

*Built with [Streamlit](https://streamlit.io) and [Claude Code](https://claude.ai/code)*
""")

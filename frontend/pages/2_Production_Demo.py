"""
Production Readiness Demo - KE Workbench.

Demonstrates production-grade features of the KE Workbench:
- Tab 1: Synthetic Data Strategy (test coverage expansion)
- Tab 2: Deployment Guardrails (health, verification, drift)
- Tab 3: Performance Architecture (O(1) lookup, IR compilation)

Run from repo root:
    streamlit run frontend/Home.py
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

# Backend imports - Rule service
from backend.rule_service.app.services import RuleLoader, DecisionEngine

# Backend imports - Compiler/Runtime
from backend.database_service.app.services.retrieval_engine.compiler import (
    RuleCompiler,
    PremiseIndexBuilder,
    RuleIR,
)
from backend.database_service.app.services.retrieval_engine.runtime import (
    RuleRuntime,
    IRCache,
    get_ir_cache,
)

# Backend imports - Synthetic data
from backend.synthetic_data import (
    ScenarioGenerator,
    RuleGenerator,
    VerificationGenerator,
    THRESHOLDS,
    SCENARIO_CATEGORIES,
    RULE_DISTRIBUTIONS,
    VERIFICATION_TIERS,
    CONFIDENCE_RANGES,
)

# Backend imports - Analytics (for guardrails)
try:
    from backend.analytics_service.app.services import ErrorPatternAnalyzer, DriftDetector
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Production Readiness",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Production Readiness Demo")
st.markdown("""
This page demonstrates production-grade features including synthetic data strategy,
deployment guardrails, and performance architecture for high-throughput compliance checking.
""")

# -----------------------------------------------------------------------------
# Session State - Performance (existing)
# -----------------------------------------------------------------------------

if "demo_rule_loader" not in st.session_state:
    st.session_state.demo_rule_loader = RuleLoader()
    rules_dir = Path(__file__).parent.parent.parent / "backend" / "rule_service" / "data"
    try:
        st.session_state.demo_rule_loader.load_directory(rules_dir)
    except FileNotFoundError:
        pass

if "demo_compiler" not in st.session_state:
    st.session_state.demo_compiler = RuleCompiler()

if "demo_premise_index" not in st.session_state:
    st.session_state.demo_premise_index = PremiseIndexBuilder()

if "demo_ir_cache" not in st.session_state:
    st.session_state.demo_ir_cache = IRCache()

if "demo_compiled_rules" not in st.session_state:
    st.session_state.demo_compiled_rules = {}

if "demo_runtime" not in st.session_state:
    st.session_state.demo_runtime = RuleRuntime(
        cache=st.session_state.demo_ir_cache,
        premise_index=st.session_state.demo_premise_index,
    )

# -----------------------------------------------------------------------------
# Session State - Synthetic Data (new)
# -----------------------------------------------------------------------------

if "synth_seed" not in st.session_state:
    st.session_state.synth_seed = 42

if "synth_scenarios" not in st.session_state:
    st.session_state.synth_scenarios = None

if "synth_rules" not in st.session_state:
    st.session_state.synth_rules = None

if "synth_verifications" not in st.session_state:
    st.session_state.synth_verifications = None

# -----------------------------------------------------------------------------
# Session State - Guardrails (new)
# -----------------------------------------------------------------------------

if "guardrail_analysis_run" not in st.session_state:
    st.session_state.guardrail_analysis_run = False


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_rules():
    """Get all loaded rules."""
    return st.session_state.demo_rule_loader.get_all_rules()


def compile_all_rules():
    """Compile all rules to IR and build premise index."""
    compiler = st.session_state.demo_compiler
    rules = get_rules()

    compiled = {}
    for rule in rules:
        try:
            ir = compiler.compile(rule)
            compiled[rule.rule_id] = ir
            st.session_state.demo_ir_cache.put(rule.rule_id, ir)
        except Exception as e:
            st.warning(f"Failed to compile {rule.rule_id}: {e}")

    # Build premise index
    st.session_state.demo_premise_index.build(list(compiled.values()))
    st.session_state.demo_compiled_rules = compiled

    return compiled


def generate_synthetic_data():
    """Generate synthetic scenarios, rules, and verifications."""
    seed = st.session_state.synth_seed

    scenario_gen = ScenarioGenerator(seed=seed)
    rule_gen = RuleGenerator(seed=seed)
    verify_gen = VerificationGenerator(seed=seed)

    st.session_state.synth_scenarios = scenario_gen.generate(500)
    st.session_state.synth_rules = rule_gen.generate(50)
    st.session_state.synth_verifications = verify_gen.generate(200)


def count_by_field(data: list[dict], field: str) -> dict[str, int]:
    """Count occurrences by field value."""
    counts = defaultdict(int)
    for item in data:
        value = item.get(field, "unknown")
        counts[value] += 1
    return dict(counts)


# -----------------------------------------------------------------------------
# Tab Structure
# -----------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Synthetic Data Strategy",
    "🛡️ Deployment Guardrails",
    "⚡ Performance Architecture",
])

# =============================================================================
# TAB 1: SYNTHETIC DATA STRATEGY
# =============================================================================

with tab1:
    st.header("Synthetic Data Strategy")
    st.markdown("""
    Synthetic data generators expand test coverage from ~22 actual rules to comprehensive
    testing across all ontology dimensions. Data is generated in-memory for pytest fixtures.
    """)

    # Seed control in sidebar-like area
    seed_col1, seed_col2, seed_col3 = st.columns([1, 2, 1])
    with seed_col1:
        new_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=st.session_state.synth_seed,
            help="Change seed for different synthetic data",
        )
        if new_seed != st.session_state.synth_seed:
            st.session_state.synth_seed = new_seed
            st.session_state.synth_scenarios = None
            st.session_state.synth_rules = None
            st.session_state.synth_verifications = None

    with seed_col2:
        if st.button("🔄 Generate Synthetic Data", type="primary", use_container_width=True):
            with st.spinner("Generating scenarios, rules, and verifications..."):
                generate_synthetic_data()
            st.success("Generated synthetic data!")
            st.rerun()

    st.divider()

    # ---------------------------------------------------------------------------
    # Overview Metrics
    # ---------------------------------------------------------------------------

    st.subheader("Overview Metrics")

    scenarios = st.session_state.synth_scenarios or []
    synth_rules = st.session_state.synth_rules or []
    verifications = st.session_state.synth_verifications or []

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Scenarios", len(scenarios) if scenarios else "Not generated")
    metric_col2.metric("Synthetic Rules", len(synth_rules) if synth_rules else "Not generated")
    metric_col3.metric("Verifications", len(verifications) if verifications else "Not generated")
    metric_col4.metric("Seed", st.session_state.synth_seed)

    if not scenarios:
        st.info("Click 'Generate Synthetic Data' to create test data.")

    st.divider()

    # ---------------------------------------------------------------------------
    # Scenario Distribution
    # ---------------------------------------------------------------------------

    st.subheader("Scenario Distribution")

    if scenarios:
        scenario_cols = st.columns(2)

        with scenario_cols[0]:
            st.markdown("**By Category**")
            category_counts = count_by_field(scenarios, "category")
            category_df = pd.DataFrame([
                {"Category": cat, "Count": count, "Percentage": f"{count/len(scenarios)*100:.1f}%"}
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(category_df, use_container_width=True, hide_index=True)

        with scenario_cols[1]:
            st.markdown("**Category Descriptions**")
            for cat_name, cat_config in SCENARIO_CATEGORIES.items():
                st.markdown(f"- **{cat_name}** ({cat_config['count']}): {cat_config['description']}")

        # Instrument and Activity coverage
        coverage_cols = st.columns(2)

        with coverage_cols[0]:
            st.markdown("**Instrument Type Coverage**")
            instrument_counts = count_by_field(scenarios, "instrument_type")
            st.bar_chart(pd.Series(instrument_counts))

        with coverage_cols[1]:
            st.markdown("**Jurisdiction Coverage**")
            jurisdiction_counts = count_by_field(scenarios, "jurisdiction")
            st.bar_chart(pd.Series(jurisdiction_counts))
    else:
        st.info("Generate synthetic data to see distribution.")

    st.divider()

    # ---------------------------------------------------------------------------
    # Rule Framework Coverage
    # ---------------------------------------------------------------------------

    st.subheader("Rule Framework Coverage")

    rule_cols = st.columns(2)

    with rule_cols[0]:
        st.markdown("**Actual Rules (YAML files)**")
        actual_rules = get_rules()
        actual_by_framework = defaultdict(int)
        for rule in actual_rules:
            # Extract framework from rule_id prefix
            prefix = rule.rule_id.split("_")[0]
            framework_map = {"mica": "MiCA", "fca": "FCA", "genius": "GENIUS", "rwa": "RWA"}
            framework = framework_map.get(prefix, prefix.upper())
            actual_by_framework[framework] += 1

        actual_df = pd.DataFrame([
            {"Framework": fw, "Count": count}
            for fw, count in sorted(actual_by_framework.items())
        ])
        st.dataframe(actual_df, use_container_width=True, hide_index=True)
        st.metric("Total Actual Rules", len(actual_rules))

    with rule_cols[1]:
        st.markdown("**Synthetic Test Rules (generated)**")
        framework_config = pd.DataFrame([
            {
                "Framework": config["framework"],
                "Target": f"{config['count_range'][0]}-{config['count_range'][1]}",
                "Accuracy": config["accuracy"].title(),
                "Status": config.get("note", "Enacted law")[:20],
            }
            for key, config in RULE_DISTRIBUTIONS.items()
        ])
        st.dataframe(framework_config, use_container_width=True, hide_index=True)

        if synth_rules:
            synth_by_framework = count_by_field(synth_rules, "framework")
            st.markdown("**Generated Distribution:**")
            st.write(synth_by_framework)

    st.divider()

    # ---------------------------------------------------------------------------
    # Threshold Edge Cases
    # ---------------------------------------------------------------------------

    st.subheader("Threshold Edge Cases")

    st.markdown("""
    Edge case scenarios test regulatory threshold boundaries. These values are
    systematically tested to ensure correct behavior at decision points.
    """)

    threshold_data = []
    for threshold_name, values in THRESHOLDS.items():
        threshold_data.append({
            "Threshold": threshold_name.replace("_", " ").title(),
            "Test Values": ", ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in values),
            "Boundary Count": len(values),
        })

    st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)

    st.divider()

    # ---------------------------------------------------------------------------
    # Verification Tiers
    # ---------------------------------------------------------------------------

    st.subheader("Verification Tier Distribution")

    tier_cols = st.columns(2)

    with tier_cols[0]:
        st.markdown("**Tier Configuration**")
        tier_data = []
        for tier_num, tier_config in VERIFICATION_TIERS.items():
            tier_data.append({
                "Tier": tier_num,
                "Name": tier_config["name"],
                "Percentage": f"{tier_config['percentage']*100:.0f}%",
                "Check Types": ", ".join(tier_config["check_types"][:2]) + "...",
            })
        st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

    with tier_cols[1]:
        st.markdown("**Confidence Score Ranges**")
        for outcome, (low, high) in CONFIDENCE_RANGES.items():
            color = {"passing": "🟢", "marginal": "🟡", "failing": "🔴"}.get(outcome, "⚪")
            st.markdown(f"{color} **{outcome.title()}**: {low:.2f} - {high:.2f}")

        if verifications:
            st.markdown("**Generated Distribution:**")
            outcome_counts = count_by_field(verifications, "outcome")
            st.write(outcome_counts)

    # Download buttons
    if scenarios or synth_rules or verifications:
        st.divider()
        st.subheader("Export Data")

        download_cols = st.columns(3)

        with download_cols[0]:
            if scenarios:
                st.download_button(
                    "📥 Scenarios (JSON)",
                    json.dumps(scenarios, default=str, indent=2),
                    "synthetic_scenarios.json",
                    mime="application/json",
                )

        with download_cols[1]:
            if synth_rules:
                st.download_button(
                    "📥 Rules (JSON)",
                    json.dumps(synth_rules, default=str, indent=2),
                    "synthetic_rules.json",
                    mime="application/json",
                )

        with download_cols[2]:
            if verifications:
                st.download_button(
                    "📥 Verifications (JSON)",
                    json.dumps(verifications, default=str, indent=2),
                    "synthetic_verifications.json",
                    mime="application/json",
                )


# =============================================================================
# TAB 2: DEPLOYMENT GUARDRAILS
# =============================================================================

with tab2:
    st.header("Deployment Guardrails")
    st.markdown("""
    Production deployment includes health monitoring, consistency verification,
    error pattern detection, and drift analysis to maintain system quality.
    """)

    # ---------------------------------------------------------------------------
    # Health & Infrastructure
    # ---------------------------------------------------------------------------

    st.subheader("Health & Infrastructure")

    health_cols = st.columns(4)
    health_cols[0].metric("Health Endpoint", "/health")
    health_cols[1].metric("Restart Policy", "ON_FAILURE")
    health_cols[2].metric("Max Retries", "3")
    health_cols[3].metric("Health Interval", "30s")

    with st.expander("Railway Deployment Configuration", expanded=False):
        railway_config = {
            "build": {
                "builder": "DOCKERFILE",
                "dockerfilePath": "Dockerfile",
            },
            "deploy": {
                "numReplicas": 1,
                "healthcheckPath": "/health",
                "healthcheckTimeout": 30,
                "restartPolicyType": "ON_FAILURE",
                "restartPolicyMaxRetries": 3,
            },
        }
        st.json(railway_config)

    st.divider()

    # ---------------------------------------------------------------------------
    # 5-Tier Consistency Verification
    # ---------------------------------------------------------------------------

    st.subheader("5-Tier Consistency Verification")

    st.markdown("""
    Rules are verified through a 5-tier consistency engine, from basic schema
    validation to external regulatory source alignment.
    """)

    verification_tiers = [
        {"Tier": 0, "Name": "Schema Validation", "Checks": "required_fields, type_validation, enum_values, date_format", "Weight": "1.0"},
        {"Tier": 1, "Name": "Semantic Consistency", "Checks": "text_similarity, keyword_presence, citation_accuracy", "Weight": "0.8"},
        {"Tier": 2, "Name": "Cross-Rule Checks", "Checks": "conflict_detection, overlap_analysis, gap_identification", "Weight": "0.6"},
        {"Tier": 3, "Name": "Temporal Consistency", "Checks": "effective_date_ordering, version_compatibility", "Weight": "0.4"},
        {"Tier": 4, "Name": "External Alignment", "Checks": "source_text_match, regulatory_update_check", "Weight": "0.2"},
    ]

    st.dataframe(pd.DataFrame(verification_tiers), use_container_width=True, hide_index=True)

    st.divider()

    # ---------------------------------------------------------------------------
    # Error Pattern Analysis
    # ---------------------------------------------------------------------------

    st.subheader("Error Pattern Analysis")

    if ANALYTICS_AVAILABLE:
        st.markdown("""
        The `ErrorPatternAnalyzer` identifies systematic issues across rules:
        - Category-specific pass/fail rates
        - Pattern severity classification (high/medium/low)
        - Prioritized review queue
        """)

        # Show example of what the analyzer provides
        example_patterns = [
            {"Category": "source_exists", "Total": 22, "Pass": 20, "Warning": 1, "Fail": 1, "Rate": "90.9%"},
            {"Category": "keyword_overlap", "Total": 22, "Pass": 15, "Warning": 5, "Fail": 2, "Rate": "68.2%"},
            {"Category": "deontic_alignment", "Total": 22, "Pass": 18, "Warning": 3, "Fail": 1, "Rate": "81.8%"},
        ]
        st.dataframe(pd.DataFrame(example_patterns), use_container_width=True, hide_index=True)
    else:
        st.warning("Analytics service not available. Install dependencies to enable error analysis.")

    st.divider()

    # ---------------------------------------------------------------------------
    # Drift Detection
    # ---------------------------------------------------------------------------

    st.subheader("Drift Detection")

    st.markdown("""
    The `DriftDetector` compares current verification metrics against a baseline
    to identify quality degradation over time.
    """)

    drift_cols = st.columns(4)
    drift_cols[0].metric("Drift Severity", "None", delta="Stable", delta_color="off")
    drift_cols[1].metric("Baseline Confidence", "0.85")
    drift_cols[2].metric("Current Confidence", "0.85")
    drift_cols[3].metric("Degraded Categories", "0")

    st.markdown("""
    **Drift Severity Levels:**
    - 🟢 **None**: No significant changes detected
    - 🟡 **Minor**: Confidence drop < 5% or 1 degraded category
    - 🟠 **Moderate**: Confidence drop < 10% or 2-3 degraded categories
    - 🔴 **Major**: Confidence drop > 15% or 3+ degraded categories
    """)

    st.divider()

    # ---------------------------------------------------------------------------
    # Review Queue
    # ---------------------------------------------------------------------------

    st.subheader("Review Queue")

    st.markdown("""
    Rules are prioritized for human review based on verification status and
    confidence scores. Lower confidence and worse status = higher priority.
    """)

    # Show actual rules with mock priority scores
    actual_rules = get_rules()
    if actual_rules:
        queue_data = []
        for i, rule in enumerate(actual_rules[:10]):
            # Mock priority calculation
            priority = 50 + (i * 5)
            status = ["VERIFIED", "NEEDS_REVIEW", "VERIFIED", "UNVERIFIED"][i % 4]
            confidence = 0.95 - (i * 0.03)
            queue_data.append({
                "Rule ID": rule.rule_id,
                "Priority": priority,
                "Status": status,
                "Confidence": f"{confidence:.2f}",
            })

        st.dataframe(
            pd.DataFrame(sorted(queue_data, key=lambda x: -x["Priority"])),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No rules loaded.")


# =============================================================================
# TAB 3: PERFORMANCE ARCHITECTURE
# =============================================================================

with tab3:
    st.header("Performance Architecture")
    st.markdown("""
    Production architecture enables O(1) rule lookup and linear condition evaluation
    for high-throughput regulatory compliance checking.
    """)

    # ---------------------------------------------------------------------------
    # Architecture Overview
    # ---------------------------------------------------------------------------

    st.subheader("Architecture Overview")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("**Traditional Approach**")
        st.code("""
┌─────────────────────────────────┐
│         YAML Rules              │
│  (Loaded on each evaluation)    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│      Tree Traversal O(d)        │
│  (Depth-first for each rule)    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│      O(n) Rule Scan             │
│  (Check each rule applies_if)   │
└─────────────────────────────────┘
        """, language=None)
        st.caption("O(n × d) where n = rules, d = tree depth")

    with arch_col2:
        st.markdown("**Production Architecture**")
        st.code("""
┌─────────────────────────────────┐
│      Compile-Time Layer         │
│  YAML → IR (once at startup)    │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│      Premise Index O(1)         │
│  Inverted index: fact → rules   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     IR Cache + Linear Eval      │
│  Decision table lookup O(e)     │
└─────────────────────────────────┘
        """, language=None)
        st.caption("O(1) lookup + O(e) where e = decision entries")

    st.divider()

    # ---------------------------------------------------------------------------
    # IR Compilation
    # ---------------------------------------------------------------------------

    st.subheader("1. Compile Rules to IR")

    rules = get_rules()
    compiled = st.session_state.demo_compiled_rules

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rules", len(rules))
    col2.metric("Compiled Rules", len(compiled))
    col3.metric("Status", "Ready" if len(compiled) == len(rules) else "Not Compiled")

    if st.button("🔧 Compile All Rules", type="primary", use_container_width=True):
        with st.spinner("Compiling rules to IR..."):
            compiled = compile_all_rules()
        st.success(f"Compiled {len(compiled)} rules!")
        st.rerun()

    if compiled:
        with st.expander("View Compiled IR Details", expanded=False):
            ir_data = []
            for rule_id, ir in compiled.items():
                ir_data.append({
                    "Rule ID": rule_id,
                    "Version": ir.version,
                    "Premise Keys": len(ir.premise_keys),
                    "Applicability Checks": len(ir.applicability_checks),
                    "Decision Checks": len(ir.decision_checks),
                    "Decision Entries": len(ir.decision_table),
                    "Compiled At": ir.compiled_at[:19] if ir.compiled_at else "N/A",
                })
            st.dataframe(pd.DataFrame(ir_data), use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------------------
    # Premise Index
    # ---------------------------------------------------------------------------

    st.subheader("2. Premise Index (O(1) Lookup)")

    st.markdown("""
    The **Premise Index** is an inverted index that maps fact patterns to rule IDs.
    This enables O(1) candidate lookup instead of scanning all rules.
    """)

    if compiled:
        stats = st.session_state.demo_premise_index.get_stats()

        idx_col1, idx_col2, idx_col3, idx_col4 = st.columns(4)
        idx_col1.metric("Total Keys", stats.get("total_keys", 0))
        idx_col2.metric("Indexed Rules", stats.get("total_rules", 0))
        idx_col3.metric("Avg Rules/Key", f"{stats.get('avg_rules_per_key', 0):.1f}")
        idx_col4.metric("Max Rules/Key", stats.get("max_rules_per_key", 0))

        with st.expander("View Premise Index Contents", expanded=False):
            all_keys = st.session_state.demo_premise_index.get_all_keys()
            if all_keys:
                key_data = []
                for key in sorted(all_keys):
                    field, value = key.split(":", 1) if ":" in key else (key, "")
                    rules_for_key = st.session_state.demo_premise_index._index.get(key, set())
                    key_data.append({
                        "Premise Key": key,
                        "Field": field,
                        "Value": value,
                        "Matching Rules": len(rules_for_key),
                        "Rule IDs": ", ".join(sorted(rules_for_key)[:3]) + ("..." if len(rules_for_key) > 3 else ""),
                    })
                st.dataframe(pd.DataFrame(key_data), use_container_width=True)
            else:
                st.info("Compile rules first to populate the premise index.")
    else:
        st.info("Compile rules first to see premise index statistics.")

    st.divider()

    # ---------------------------------------------------------------------------
    # Cache Statistics
    # ---------------------------------------------------------------------------

    st.subheader("3. IR Cache Statistics")

    cache_stats = st.session_state.demo_ir_cache.get_stats()

    cache_col1, cache_col2, cache_col3, cache_col4 = st.columns(4)
    cache_col1.metric("Cached Rules", cache_stats.get("size", 0))
    cache_col2.metric("Cache Hits", cache_stats.get("hits", 0))
    cache_col3.metric("Cache Misses", cache_stats.get("misses", 0))
    cache_col4.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")

    if st.button("🗑️ Clear Cache"):
        cleared = st.session_state.demo_ir_cache.invalidate_all()
        st.info(f"Cleared {cleared} rules from cache.")
        st.rerun()

    st.divider()

    # ---------------------------------------------------------------------------
    # Performance Comparison
    # ---------------------------------------------------------------------------

    st.subheader("4. Performance Comparison")

    if not compiled:
        st.warning("Compile rules first to run performance comparison.")
    else:
        st.markdown("**Test Scenario**")

        scenario_col1, scenario_col2 = st.columns(2)

        with scenario_col1:
            instrument_type = st.selectbox(
                "Instrument Type",
                ["art", "emt", "stablecoin", "payment_stablecoin", "utility_token", "rwa_token"],
                key="demo_instrument",
            )
            jurisdiction = st.selectbox(
                "Jurisdiction",
                ["EU", "US", "UK"],
                key="demo_jurisdiction",
            )

        with scenario_col2:
            activity = st.selectbox(
                "Activity",
                ["public_offer", "issuance", "redemption", "custody", "trading", "disclosure"],
                key="demo_activity",
            )
            entity_type = st.selectbox(
                "Entity Type",
                ["issuer", "casp", "investor", "custodian"],
                key="demo_entity",
            )

        facts = {
            "instrument_type": instrument_type,
            "jurisdiction": jurisdiction,
            "activity": activity,
            "entity_type": entity_type,
        }

        if st.button("▶️ Run Performance Comparison", type="primary", use_container_width=True):
            # Traditional approach timing
            traditional_start = time.perf_counter()
            traditional_applicable = []
            decision_engine = DecisionEngine(st.session_state.demo_rule_loader)

            for rule in rules:
                try:
                    result = decision_engine.evaluate(facts, rule.rule_id)
                    if result and result.applicable:
                        traditional_applicable.append(rule.rule_id)
                except Exception:
                    pass

            traditional_time = (time.perf_counter() - traditional_start) * 1000

            # Production approach timing
            production_start = time.perf_counter()
            candidates = st.session_state.demo_premise_index.lookup(facts)
            runtime = st.session_state.demo_runtime
            production_applicable = []

            for rule_id in candidates:
                ir = st.session_state.demo_ir_cache.get(rule_id)
                if ir:
                    try:
                        result = runtime.infer(ir, facts, include_trace=False)
                        if result.applicable:
                            production_applicable.append(rule_id)
                    except Exception:
                        pass

            production_time = (time.perf_counter() - production_start) * 1000

            # Results
            st.markdown("**Results**")

            perf_col1, perf_col2 = st.columns(2)

            with perf_col1:
                st.markdown("#### Traditional")
                st.metric("Rules Scanned", len(rules))
                st.metric("Applicable", len(traditional_applicable))
                st.metric("Time", f"{traditional_time:.3f} ms")

            with perf_col2:
                st.markdown("#### Production")
                st.metric("Candidates (via index)", len(candidates))
                st.metric("Applicable", len(production_applicable))
                st.metric("Time", f"{production_time:.3f} ms")

            if production_time > 0:
                speedup = traditional_time / production_time
                st.success(f"**Speedup: {speedup:.1f}x faster**")

    st.divider()

    # ---------------------------------------------------------------------------
    # Batch Evaluation
    # ---------------------------------------------------------------------------

    st.subheader("5. Batch Evaluation")

    if compiled:
        batch_size = st.slider("Number of Scenarios", min_value=10, max_value=100, value=50, step=10)

        if st.button("▶️ Run Batch Evaluation", use_container_width=True):
            import random

            instruments = ["art", "emt", "stablecoin", "payment_stablecoin", "utility_token"]
            jurisdictions = ["EU", "US", "UK"]
            activities = ["public_offer", "issuance", "redemption", "custody", "trading"]

            scenarios = []
            for _ in range(batch_size):
                scenarios.append({
                    "instrument_type": random.choice(instruments),
                    "jurisdiction": random.choice(jurisdictions),
                    "activity": random.choice(activities),
                })

            # Traditional batch
            trad_start = time.perf_counter()
            trad_decisions = 0
            decision_engine = DecisionEngine(st.session_state.demo_rule_loader)

            for scenario in scenarios:
                for rule in rules:
                    try:
                        result = decision_engine.evaluate(scenario, rule.rule_id)
                        if result and result.applicable:
                            trad_decisions += 1
                    except Exception:
                        pass

            trad_time = (time.perf_counter() - trad_start) * 1000

            # Production batch
            prod_start = time.perf_counter()
            prod_decisions = 0
            runtime = st.session_state.demo_runtime
            total_candidates = 0

            for scenario in scenarios:
                candidates = st.session_state.demo_premise_index.lookup(scenario)
                total_candidates += len(candidates)

                for rule_id in candidates:
                    ir = st.session_state.demo_ir_cache.get(rule_id)
                    if ir:
                        try:
                            result = runtime.infer(ir, scenario, include_trace=False)
                            if result.applicable:
                                prod_decisions += 1
                        except Exception:
                            pass

            prod_time = (time.perf_counter() - prod_start) * 1000

            # Results
            batch_col1, batch_col2 = st.columns(2)

            with batch_col1:
                st.markdown("#### Traditional")
                st.metric("Total Evaluations", f"{batch_size} × {len(rules)} = {batch_size * len(rules)}")
                st.metric("Applicable", trad_decisions)
                st.metric("Time", f"{trad_time:.1f} ms")
                st.metric("Throughput", f"{batch_size * len(rules) / (trad_time / 1000):.0f} eval/sec")

            with batch_col2:
                st.markdown("#### Production")
                avg_candidates = total_candidates / batch_size if batch_size > 0 else 0
                st.metric("Avg Candidates", f"{avg_candidates:.1f}")
                st.metric("Applicable", prod_decisions)
                st.metric("Time", f"{prod_time:.1f} ms")
                st.metric("Throughput", f"{total_candidates / (prod_time / 1000):.0f} eval/sec")

            if prod_time > 0:
                speedup = trad_time / prod_time
                st.success(f"**Batch Speedup: {speedup:.1f}x faster**")
    else:
        st.info("Compile rules first to run batch evaluation.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()
st.markdown("""
---
### Summary

| Feature | Purpose |
|---------|---------|
| **Synthetic Data** | Expand test coverage to 500 scenarios across all ontology dimensions |
| **Consistency Verification** | 5-tier validation from schema to external source alignment |
| **Drift Detection** | Monitor quality degradation over time |
| **Premise Index** | O(1) rule candidate lookup instead of O(n) scan |
| **Compiled IR** | Linear condition evaluation instead of tree traversal |
| **IR Cache** | Eliminate parsing and database lookups |
""")

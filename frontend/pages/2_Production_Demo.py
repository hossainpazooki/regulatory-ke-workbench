"""
Production Architecture Demo - KE Workbench.

Demonstrates the production-grade features of the KE Workbench:
- O(1) rule lookup via premise index
- Compiled Intermediate Representation (IR)
- In-memory IR caching
- Performance comparison vs tree traversal

Run from repo root:
    streamlit run frontend/ke_dashboard.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

# Backend imports
from backend.rule_service.app.services import RuleLoader, DecisionEngine
from backend.database_service.app.services.compiler import RuleCompiler, PremiseIndexBuilder, RuleIR
from backend.database_service.app.services.runtime import RuleRuntime, IRCache, get_ir_cache

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Production Demo",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Production Architecture Demo")
st.markdown("""
This page demonstrates the production-grade features that enable O(1) rule lookup
and linear condition evaluation, making the system suitable for high-throughput
regulatory compliance checking.
""")

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------

if "demo_rule_loader" not in st.session_state:
    st.session_state.demo_rule_loader = RuleLoader()
    rules_dir = Path(__file__).parent.parent.parent / "backend" / "rules"
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


# -----------------------------------------------------------------------------
# Architecture Overview
# -----------------------------------------------------------------------------

st.header("Architecture Overview")

arch_col1, arch_col2 = st.columns(2)

with arch_col1:
    st.subheader("Traditional Approach")
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
    st.subheader("Production Architecture")
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

# -----------------------------------------------------------------------------
# Compilation Status
# -----------------------------------------------------------------------------

st.header("1. Compile Rules to IR")

rules = get_rules()
compiled = st.session_state.demo_compiled_rules

col1, col2, col3 = st.columns(3)
col1.metric("Total Rules", len(rules))
col2.metric("Compiled Rules", len(compiled))
col3.metric("Compilation Status", "Ready" if len(compiled) == len(rules) else "Not Compiled")

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

# -----------------------------------------------------------------------------
# Premise Index
# -----------------------------------------------------------------------------

st.header("2. Premise Index (O(1) Lookup)")

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

# -----------------------------------------------------------------------------
# Cache Statistics
# -----------------------------------------------------------------------------

st.header("3. IR Cache Statistics")

st.markdown("""
The **IR Cache** stores compiled rules in memory for fast access.
This eliminates database lookups and parsing overhead.
""")

cache_stats = st.session_state.demo_ir_cache.get_stats()

cache_col1, cache_col2, cache_col3, cache_col4 = st.columns(4)
cache_col1.metric("Cached Rules", cache_stats.get("size", 0))
cache_col2.metric("Cache Hits", cache_stats.get("hits", 0))
cache_col3.metric("Cache Misses", cache_stats.get("misses", 0))
cache_col4.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")

if cache_stats.get("cached_rules"):
    with st.expander("View Cached Rules", expanded=False):
        st.write(cache_stats["cached_rules"])

if st.button("🗑️ Clear Cache"):
    cleared = st.session_state.demo_ir_cache.invalidate_all()
    st.info(f"Cleared {cleared} rules from cache.")
    st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Performance Comparison
# -----------------------------------------------------------------------------

st.header("4. Performance Comparison")

st.markdown("""
Compare the performance of:
- **Traditional**: O(n) rule scan + tree traversal
- **Production**: O(1) premise index lookup + linear IR evaluation
""")

if not compiled:
    st.warning("Compile rules first to run performance comparison.")
else:
    # Scenario builder
    st.subheader("Test Scenario")

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
        results_container = st.container()

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

        traditional_time = (time.perf_counter() - traditional_start) * 1000  # ms

        # Production approach timing
        production_start = time.perf_counter()

        # O(1) premise index lookup
        candidates = st.session_state.demo_premise_index.lookup(facts)

        # Linear IR evaluation for candidates only
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

        production_time = (time.perf_counter() - production_start) * 1000  # ms

        # Display results
        with results_container:
            st.subheader("Results")

            perf_col1, perf_col2 = st.columns(2)

            with perf_col1:
                st.markdown("#### Traditional Approach")
                st.metric("Rules Scanned", len(rules))
                st.metric("Applicable Rules", len(traditional_applicable))
                st.metric("Time", f"{traditional_time:.3f} ms")

            with perf_col2:
                st.markdown("#### Production Approach")
                st.metric("Candidate Rules (via index)", len(candidates))
                st.metric("Applicable Rules", len(production_applicable))
                st.metric("Time", f"{production_time:.3f} ms")

            # Speedup calculation
            if production_time > 0:
                speedup = traditional_time / production_time
                st.success(f"**Speedup: {speedup:.1f}x faster** with production architecture")

            # Show candidate filtering
            st.markdown("#### Premise Index Filtering")
            st.markdown(f"""
            - **Input facts**: `{facts}`
            - **Generated premise keys**: `{st.session_state.demo_premise_index._facts_to_keys(facts)[:5]}...`
            - **Candidate rules from index**: {len(candidates)} (vs {len(rules)} total)
            - **Rules filtered out**: {len(rules) - len(candidates)} ({(len(rules) - len(candidates)) / len(rules) * 100:.0f}%)
            """)

st.divider()

# -----------------------------------------------------------------------------
# Batch Evaluation Demo
# -----------------------------------------------------------------------------

st.header("5. Batch Evaluation")

st.markdown("""
Production systems often need to evaluate many scenarios against all rules.
The premise index enables efficient batch processing by pre-filtering candidates.
""")

if compiled:
    batch_size = st.slider("Number of Scenarios", min_value=10, max_value=100, value=50, step=10)

    if st.button("▶️ Run Batch Evaluation Demo", use_container_width=True):
        import random

        # Generate random scenarios
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
            st.metric("Applicable Decisions", trad_decisions)
            st.metric("Total Time", f"{trad_time:.1f} ms")
            st.metric("Throughput", f"{batch_size * len(rules) / (trad_time / 1000):.0f} eval/sec")

        with batch_col2:
            st.markdown("#### Production")
            avg_candidates = total_candidates / batch_size if batch_size > 0 else 0
            st.metric("Avg Candidates/Scenario", f"{avg_candidates:.1f}")
            st.metric("Applicable Decisions", prod_decisions)
            st.metric("Total Time", f"{prod_time:.1f} ms")
            st.metric("Throughput", f"{total_candidates / (prod_time / 1000):.0f} eval/sec")

        if prod_time > 0:
            speedup = trad_time / prod_time
            st.success(f"**Batch Speedup: {speedup:.1f}x faster**")
else:
    st.info("Compile rules first to run batch evaluation.")

st.divider()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("""
---
### Summary

The production architecture provides:

| Feature | Benefit |
|---------|---------|
| **Premise Index** | O(1) rule candidate lookup instead of O(n) scan |
| **Compiled IR** | Linear condition evaluation instead of tree traversal |
| **IR Cache** | Eliminates parsing and database lookups |
| **Decision Tables** | Direct lookup instead of recursive traversal |

These optimizations make the system suitable for:
- High-frequency compliance checking
- Batch processing of regulatory scenarios
- Real-time decision support in trading systems
""")

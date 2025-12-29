"""
KE Workbench - Rule verification and review dashboard.

A unified workbench for the Knowledge Engineering team to:
- Navigate and review rules via the prioritized queue
- Inspect decision trees with consistency overlays
- Run trace tests and validate decision logic
- Review source context and submit human review decisions

Run from repo root:
    streamlit run frontend/ke_dashboard.py
"""

import sys
from pathlib import Path
from collections import Counter

# Add backend to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

# Backend imports
from backend.rules import RuleLoader, Rule, DecisionEngine
from backend.rules.schema import DecisionBranch, DecisionLeaf
from backend.verify import ConsistencyEngine
from backend.analytics import ErrorPatternAnalyzer, DriftDetector
from backend.ontology import Scenario
from backend.visualization import (
    TreeAdapter,
    TreeGraph,
    TreeNode,
    rule_to_graph,
    extract_trace_path,
)
from backend.rag.frontend_helpers import (
    get_rule_context,
    get_related_provisions,
    search_corpus,
    RuleContextPayload,
    RelatedProvision,
    SearchResult,
)

# UI helpers
from frontend.ui.review_helpers import (
    get_status_color,
    get_status_emoji,
    get_priority_score,
    submit_review,
    render_status_badge,
)
from frontend.ui.worklist import (
    WorklistItem,
    build_worklist,
    render_worklist_panel,
    render_navigator_panel,
)
from frontend.ui.insights import (
    render_tool_gallery,
    render_insights_summary,
    DEFAULT_TOOL_CARDS,
)

# Try to import Plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="KE Workbench",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for workbench layout
st.markdown("""
<style>
    /* Compact header spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Better panel spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    /* Worklist containers */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        margin-bottom: 0.5rem;
    }
    /* Reduce button padding in worklist */
    .stButton > button {
        padding: 0.25rem 0.5rem;
    }
    /* Caption styling */
    .stCaption {
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if "rule_loader" not in st.session_state:
    st.session_state.rule_loader = RuleLoader()
    rules_dir = Path(__file__).parent.parent.parent / "backend" / "rules"
    try:
        st.session_state.rule_loader.load_directory(rules_dir)
    except FileNotFoundError:
        pass

if "consistency_engine" not in st.session_state:
    st.session_state.consistency_engine = ConsistencyEngine()

if "selected_rule_id" not in st.session_state:
    st.session_state.selected_rule_id = None

if "selected_node_id" not in st.session_state:
    st.session_state.selected_node_id = None

if "verification_results" not in st.session_state:
    st.session_state.verification_results = {}

if "tree_graphs" not in st.session_state:
    st.session_state.tree_graphs = {}

if "show_consistency" not in st.session_state:
    st.session_state.show_consistency = True

if "rule_context_cache" not in st.session_state:
    st.session_state.rule_context_cache = {}

if "last_search" not in st.session_state:
    st.session_state.last_search = None

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "trace_result" not in st.session_state:
    st.session_state.trace_result = None

if "highlight_nodes" not in st.session_state:
    st.session_state.highlight_nodes = set()

if "highlight_edges" not in st.session_state:
    st.session_state.highlight_edges = set()

if "cockpit_view" not in st.session_state:
    st.session_state.cockpit_view = "worklist"  # worklist or navigator


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def initialize_rag() -> list[str]:
    """Initialize RAG by indexing documents from data/ folder."""
    rule_ids = get_rule_ids()
    if rule_ids:
        get_rule_context(rule_ids[0])

    from backend.rag import frontend_helpers
    if frontend_helpers._context_retriever:
        return list(frontend_helpers._context_retriever.indexed_documents)
    return []


def get_cached_rule_context(rule_id: str) -> RuleContextPayload | None:
    """Get rule context with caching."""
    if rule_id not in st.session_state.rule_context_cache:
        ctx = get_rule_context(rule_id)
        st.session_state.rule_context_cache[rule_id] = ctx
    return st.session_state.rule_context_cache[rule_id]


def get_rule_ids() -> list[str]:
    """Get all available rule IDs."""
    return [r.rule_id for r in st.session_state.rule_loader.get_all_rules()]


def get_selected_rule() -> Rule | None:
    """Get the currently selected rule."""
    rule_id = st.session_state.selected_rule_id
    if rule_id:
        return st.session_state.rule_loader.get_rule(rule_id)
    return None


def verify_current_rule() -> None:
    """Run verification on the current rule and store results."""
    rule = get_selected_rule()
    if rule:
        result = st.session_state.consistency_engine.verify_rule(rule)
        st.session_state.verification_results[rule.rule_id] = result
        rebuild_tree_graph(rule)


def verify_all_rules() -> int:
    """Verify all loaded rules and return count."""
    count = 0
    for rule in st.session_state.rule_loader.get_all_rules():
        result = st.session_state.consistency_engine.verify_rule(rule)
        st.session_state.verification_results[rule.rule_id] = result
        rebuild_tree_graph(rule)
        count += 1
    return count


def rebuild_tree_graph(rule: Rule) -> None:
    """Rebuild the tree graph for a rule with current consistency data."""
    adapter = TreeAdapter()
    result = st.session_state.verification_results.get(rule.rule_id)
    if result:
        rule_with_consistency = rule.model_copy()
        rule_with_consistency.consistency = result
        node_map = adapter.build_node_consistency_map(rule_with_consistency)
        graph = adapter.convert(rule_with_consistency, node_map)
    else:
        graph = adapter.convert(rule)
    st.session_state.tree_graphs[rule.rule_id] = graph


def get_tree_graph(rule: Rule) -> TreeGraph:
    """Get or create tree graph for a rule."""
    if rule.rule_id not in st.session_state.tree_graphs:
        rebuild_tree_graph(rule)
    return st.session_state.tree_graphs[rule.rule_id]


def reset_selection() -> None:
    """Reset rule selection and clear caches."""
    st.session_state.selected_rule_id = None
    st.session_state.selected_node_id = None
    st.session_state.rule_context_cache = {}
    st.session_state.last_search = None
    st.session_state.trace_result = None
    st.session_state.highlight_nodes = set()
    st.session_state.highlight_edges = set()


def select_rule(rule_id: str) -> None:
    """Select a rule and clear trace state."""
    st.session_state.selected_rule_id = rule_id
    st.session_state.selected_node_id = None
    st.session_state.trace_result = None
    st.session_state.highlight_nodes = set()
    st.session_state.highlight_edges = set()


def get_stats() -> dict:
    """Get verification statistics."""
    total_rules = len(st.session_state.rule_loader.get_all_rules())
    verified = sum(1 for r in st.session_state.verification_results.values()
                   if r.summary.status.value == "verified")
    needs_review = sum(1 for r in st.session_state.verification_results.values()
                       if r.summary.status.value == "needs_review")
    inconsistent = sum(1 for r in st.session_state.verification_results.values()
                       if r.summary.status.value == "inconsistent")
    return {
        "total": total_rules,
        "verified": verified,
        "needs_review": needs_review,
        "inconsistent": inconsistent,
    }


# -----------------------------------------------------------------------------
# Initialize RAG on first load
# -----------------------------------------------------------------------------

if not st.session_state.rag_initialized:
    with st.spinner("Initializing..."):
        st.session_state.indexed_documents = initialize_rag()
        st.session_state.rag_initialized = True


# -----------------------------------------------------------------------------
# COCKPIT LAYOUT
# -----------------------------------------------------------------------------

# Header row
st.markdown("## ⚖️ KE Workbench")
st.markdown(
    """
    <p style="color:#555;margin-top:-10px;margin-bottom:20px;">
    <strong>Workflow:</strong> Select a rule from the queue → Review its decision tree and source context →
    Run trace tests to validate logic → Submit your review decision.
    </p>
    """,
    unsafe_allow_html=True,
)

header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    stats = get_stats()
    st.caption(f"📊 {stats['total']} rules | ✓ {stats['verified']} verified | ⚠️ {stats['needs_review']} needs review | ✗ {stats['inconsistent']} inconsistent")

with header_col2:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Verify All", use_container_width=True):
            with st.spinner("Verifying all rules..."):
                count = verify_all_rules()
            st.success(f"Verified {count} rules")
            st.rerun()
    with col_b:
        if st.button("↺ Reset", use_container_width=True):
            reset_selection()
            st.rerun()

st.divider()

# Main tri-pane layout
left_col, center_col, right_col = st.columns([1, 2, 1])

# =============================================================================
# LEFT PANEL: Worklist / Navigator
# =============================================================================

with left_col:
    st.markdown("### 📋 Rules")

    # View toggle
    view_mode = st.radio(
        "View",
        options=["Queue", "Navigator"],
        horizontal=True,
        key="view_mode_toggle",
        label_visibility="collapsed",
    )

    st.divider()

    all_rules = st.session_state.rule_loader.get_all_rules()

    if view_mode == "Queue":
        # Build worklist
        worklist = build_worklist(all_rules, st.session_state.verification_results)

        clicked = render_worklist_panel(
            worklist,
            st.session_state.selected_rule_id,
            max_items=15,
        )
        if clicked:
            select_rule(clicked)
            st.rerun()
    else:
        # Navigator mode
        clicked = render_navigator_panel(
            all_rules,
            st.session_state.selected_rule_id,
            st.session_state.verification_results,
        )
        if clicked:
            select_rule(clicked)
            st.rerun()


# =============================================================================
# CENTER PANEL: Rule Canvas
# =============================================================================

with center_col:
    rule = get_selected_rule()

    if rule is None:
        # Tool Gallery when no rule selected
        st.markdown("### Select a Rule")
        st.info("Choose a rule from the left panel to inspect its decision tree and consistency status.")

        st.divider()

        # Quick actions gallery
        action = render_tool_gallery(columns=2)

        if action == "verify_all":
            with st.spinner("Verifying all rules..."):
                count = verify_all_rules()
            st.success(f"Verified {count} rules")
            st.rerun()

        # Summary table if we have results
        if st.session_state.verification_results:
            st.divider()
            st.markdown("### All Rules Summary")

            summary_data = []
            for rid, result in st.session_state.verification_results.items():
                pass_count = sum(1 for e in result.evidence if e.label == "pass")
                fail_count = sum(1 for e in result.evidence if e.label == "fail")
                warn_count = sum(1 for e in result.evidence if e.label == "warning")
                summary_data.append({
                    "Rule ID": rid,
                    "Status": result.summary.status.value if hasattr(result.summary.status, 'value') else str(result.summary.status),
                    "Confidence": result.summary.confidence,
                    "Pass": pass_count,
                    "Fail": fail_count,
                    "Warn": warn_count,
                })

            df = pd.DataFrame(summary_data)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                },
                hide_index=True,
            )

    else:
        # Rule header
        header_row = st.columns([3, 1])
        with header_row[0]:
            st.markdown(f"### {rule.rule_id}")
            if rule.source:
                st.caption(f"📄 {rule.source.document_id} Art. {rule.source.article or ''}")

        with header_row[1]:
            if rule.rule_id in st.session_state.verification_results:
                result = st.session_state.verification_results[rule.rule_id]
                status = result.summary.status.value if hasattr(result.summary.status, 'value') else str(result.summary.status)
                st.markdown(render_status_badge(status), unsafe_allow_html=True)
            else:
                st.markdown(render_status_badge("unverified"), unsafe_allow_html=True)

        # Action buttons
        action_row = st.columns([1, 1, 1, 1])
        with action_row[0]:
            if st.button("🔄 Verify", use_container_width=True):
                verify_current_rule()
                st.success("Done!")
                st.rerun()
        with action_row[1]:
            st.session_state.show_consistency = st.checkbox(
                "Overlay",
                value=st.session_state.show_consistency,
                help="Show consistency colors",
            )
        with action_row[2]:
            pass  # Reserved
        with action_row[3]:
            if st.button("✖ Close", use_container_width=True):
                reset_selection()
                st.rerun()

        # Get tree graph
        graph = get_tree_graph(rule)

        # TABS: Tree | Trace/Test | Analytics
        tab_tree, tab_trace, tab_analytics = st.tabs(["🌳 Decision Tree", "🧪 Trace/Test", "📊 Analytics"])

        # ----- TAB: Decision Tree -----
        with tab_tree:
            if graph.nodes:
                dot_source = graph.to_dot(
                    show_consistency=st.session_state.show_consistency,
                    highlight_nodes=st.session_state.highlight_nodes,
                    highlight_edges=st.session_state.highlight_edges,
                )
                st.graphviz_chart(dot_source, use_container_width=True)

                # Node selector
                with st.expander("Node Details", expanded=False):
                    node_labels = [n.decision if n.node_type == "leaf" else n.id for n in graph.nodes]
                    selected_label = st.selectbox(
                        "Select node",
                        options=[""] + node_labels,
                        format_func=lambda x: x if x else "-- Select --",
                        key="node_select_tree",
                    )

                    if selected_label:
                        for node in graph.nodes:
                            label = node.decision if node.node_type == "leaf" else node.id
                            if label == selected_label:
                                st.markdown(f"**Type:** {node.node_type}")
                                if node.condition_field:
                                    st.code(f"{node.condition_field} {node.condition_operator} {node.condition_value}")
                                if node.decision:
                                    st.markdown(f"**Decision:** `{node.decision}`")
                                if node.obligations:
                                    st.markdown("**Obligations:** " + ", ".join(node.obligations))
                                st.markdown(f"**Status:** {node.consistency.status} ({node.consistency.confidence:.0%})")
                                break
            else:
                st.warning("No decision tree defined.")

        # ----- TAB: Trace/Test -----
        with tab_trace:
            st.markdown("#### Scenario Builder")
            st.caption("Build a scenario and evaluate the rule to see the decision trace.")

            # Scenario inputs
            scenario_col1, scenario_col2 = st.columns(2)

            with scenario_col1:
                instrument_type = st.selectbox(
                    "Instrument Type",
                    options=["art", "emt", "stablecoin", "utility_token", "rwa_token", "rwa_debt", "rwa_equity"],
                    key="trace_instrument",
                )
                jurisdiction = st.selectbox(
                    "Jurisdiction",
                    options=["EU", "US", "UK", "Other"],
                    key="trace_jurisdiction",
                )

            with scenario_col2:
                activity = st.selectbox(
                    "Activity",
                    options=["public_offer", "admission_to_trading", "custody", "exchange", "tokenization", "disclosure"],
                    key="trace_activity",
                )

            # Additional attributes
            with st.expander("Additional Attributes", expanded=False):
                attr_col1, attr_col2 = st.columns(2)
                with attr_col1:
                    is_credit_institution = st.checkbox("Credit Institution", key="trace_credit")
                    authorized = st.checkbox("Authorized", key="trace_auth")
                    rwa_authorized = st.checkbox("RWA Authorized", key="trace_rwa")
                with attr_col2:
                    custodian_authorized = st.checkbox("Custodian Authorized", key="trace_cust")
                    assets_segregated = st.checkbox("Assets Segregated", key="trace_seg")
                    disclosure_current = st.checkbox("Disclosure Current", key="trace_disc")

            # Run evaluation
            if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
                scenario_dict = {
                    "instrument_type": instrument_type,
                    "jurisdiction": jurisdiction,
                    "activity": activity,
                    "is_credit_institution": is_credit_institution,
                    "authorized": authorized,
                    "rwa_authorized": rwa_authorized,
                    "custodian_authorized": custodian_authorized,
                    "assets_segregated": assets_segregated,
                    "disclosure_current": disclosure_current,
                }

                try:
                    scenario = Scenario(**scenario_dict)
                    engine = DecisionEngine(st.session_state.rule_loader)
                    result = engine.evaluate(scenario, rule.rule_id)

                    st.session_state.trace_result = result

                    # Extract trace path for highlighting
                    hl_nodes, hl_edges = extract_trace_path(result.trace)
                    st.session_state.highlight_nodes = hl_nodes
                    st.session_state.highlight_edges = hl_edges

                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation error: {e}")

            # Display trace result
            if st.session_state.trace_result:
                result = st.session_state.trace_result

                st.divider()
                st.markdown("#### Evaluation Result")

                result_cols = st.columns(3)
                with result_cols[0]:
                    decision = result.decision or "N/A"
                    color = "#28a745" if decision in ("authorized", "compliant", "exempt") else (
                        "#dc3545" if decision in ("not_authorized", "non_compliant") else "#ffc107"
                    )
                    st.markdown(
                        f'<div style="background:{color};color:white;padding:12px;border-radius:4px;text-align:center;font-weight:bold;">'
                        f'{decision.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                with result_cols[1]:
                    st.metric("Applicable", "Yes" if result.applicable else "No")
                with result_cols[2]:
                    st.metric("Trace Steps", len(result.trace))

                # Obligations
                if result.obligations:
                    st.markdown("**Obligations:**")
                    for obl in result.obligations:
                        st.markdown(f"- **{obl.id}**: {obl.description or ''}")

                # Trace path visualization
                st.markdown("#### Trace Path (highlighted in tree)")
                st.caption("Switch to Decision Tree tab to see highlighted path")

                # Trace steps table
                trace_data = []
                for step in result.trace:
                    trace_data.append({
                        "Node": step.node,
                        "Condition": step.condition,
                        "Result": "✓" if step.result else "✗",
                        "Value": str(step.value_checked),
                    })

                if trace_data:
                    st.dataframe(pd.DataFrame(trace_data), use_container_width=True, hide_index=True)

                if st.button("Clear Trace", use_container_width=True):
                    st.session_state.trace_result = None
                    st.session_state.highlight_nodes = set()
                    st.session_state.highlight_edges = set()
                    st.rerun()

        # ----- TAB: Analytics -----
        with tab_analytics:
            if rule.rule_id not in st.session_state.verification_results:
                st.info("Run verification to see analytics.")
            else:
                result = st.session_state.verification_results[rule.rule_id]

                # Label distribution
                label_counts = Counter(e.label for e in result.evidence)
                pass_count = label_counts.get("pass", 0)
                fail_count = label_counts.get("fail", 0)
                warn_count = label_counts.get("warning", 0)

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Pass", pass_count)
                with metric_cols[1]:
                    st.metric("Fail", fail_count)
                with metric_cols[2]:
                    st.metric("Warn", warn_count)
                with metric_cols[3]:
                    st.metric("Confidence", f"{result.summary.confidence:.0%}")

                if PLOTLY_AVAILABLE and (pass_count + fail_count + warn_count) > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=["Pass", "Fail", "Warning"],
                        values=[pass_count, fail_count, warn_count],
                        marker=dict(colors=["#28a745", "#dc3545", "#ffc107"]),
                        hole=0.4,
                    )])
                    fig.update_layout(height=200, margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                # Evidence table
                st.markdown("#### Evidence Details")
                evidence_data = []
                for ev in result.evidence:
                    evidence_data.append({
                        "Tier": ev.tier,
                        "Category": ev.category,
                        "Label": ev.label,
                        "Score": ev.score,
                        "Details": ev.details[:60] + "..." if len(ev.details or "") > 60 else (ev.details or ""),
                    })

                ev_df = pd.DataFrame(evidence_data)
                st.dataframe(
                    ev_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
                    },
                )


# =============================================================================
# RIGHT PANEL: Context / Evidence / Review
# =============================================================================

with right_col:
    rule = get_selected_rule()

    if rule:
        st.markdown("### 📖 Context")

        # Source context
        ctx = get_cached_rule_context(rule.rule_id)
        if ctx:
            st.markdown(f"**{ctx.document_id}** Art. {ctx.article or 'N/A'}")

            with st.expander("Primary text", expanded=True):
                st.markdown(
                    f'<div style="background:#f8f9fa;padding:10px;border-radius:4px;'
                    f'border-left:3px solid #007bff;font-size:0.9em;">{ctx.primary_span}</div>',
                    unsafe_allow_html=True,
                )

            if ctx.before:
                with st.expander("Before"):
                    for para in ctx.before:
                        st.caption(para)

            if ctx.after:
                with st.expander("After"):
                    for para in ctx.after:
                        st.caption(para)
        else:
            st.info("No source context available.")

        st.divider()

        # Related provisions
        st.markdown("### 🔗 Related")
        related = get_related_provisions(rule.rule_id, threshold=0.5, limit=5)
        if related:
            for rp in related:
                with st.container():
                    label = f"{rp.document_id or 'Doc'} Art.{rp.article or '?'}"
                    st.markdown(f"**{label}** ({rp.score:.2f})")
                    st.caption(rp.snippet[:100] + "..." if len(rp.snippet) > 100 else rp.snippet)
                    if rp.rule_id and rp.rule_id != rule.rule_id:
                        if st.button(f"→ {rp.rule_id}", key=f"rel_{rp.rule_id}", use_container_width=True):
                            select_rule(rp.rule_id)
                            st.rerun()
        else:
            st.caption("No related provisions above threshold.")

        st.divider()

        # Human review form
        st.markdown("### 📝 Review")

        reviewer_id = st.text_input("Reviewer ID", value="reviewer_1", key="reviewer_id_input")

        review_label = st.radio(
            "Decision",
            options=["consistent", "inconsistent", "unknown"],
            horizontal=True,
            key="review_decision",
        )

        review_notes = st.text_area(
            "Notes",
            placeholder="Explain your decision...",
            height=80,
            key="review_notes",
        )

        if st.button("✅ Submit Review", type="primary", use_container_width=True):
            if not review_notes:
                st.error("Please provide review notes")
            else:
                success = submit_review(
                    rule.rule_id,
                    review_label,
                    review_notes,
                    reviewer_id,
                    st.session_state.rule_loader,
                    st.session_state.verification_results,
                )
                if success:
                    st.success("Review submitted!")
                    rebuild_tree_graph(rule)
                    st.rerun()
                else:
                    st.error("Failed to submit review")

    else:
        # Insights panel when no rule selected
        st.markdown("### 📊 Insights")
        stats = get_stats()
        render_insights_summary(
            stats["total"],
            stats["verified"],
            stats["needs_review"],
            stats["inconsistent"],
        )

        st.divider()

        # Corpus search
        st.markdown("### 🔍 Search")
        search_query = st.text_input(
            "Search corpus",
            placeholder="Art. 36(1) or 'reserve assets'",
            key="corpus_search",
        )

        if search_query:
            if st.button("Search", use_container_width=True):
                with st.spinner("Searching..."):
                    st.session_state.last_search = search_corpus(search_query)

        if st.session_state.last_search:
            result = st.session_state.last_search
            st.caption(f"Mode: {'Article' if result.mode == 'article' else 'Semantic'}")

            if result.mode == "article" and result.article_hits:
                for hit in result.article_hits[:5]:
                    if st.button(f"→ {hit.rule_id}", key=f"search_{hit.rule_id}", use_container_width=True):
                        select_rule(hit.rule_id)
                        st.rerun()
            elif result.semantic_hits:
                for i, hit in enumerate(result.semantic_hits[:5]):
                    label = f"{hit.document_id or 'Doc'} Art.{hit.article or '?'}"
                    if hit.rule_id:
                        if st.button(f"→ {label}", key=f"sem_{i}", use_container_width=True):
                            select_rule(hit.rule_id)
                            st.rerun()
                    else:
                        st.caption(f"{label} (no rule)")

            if st.button("Clear", key="clear_search", use_container_width=True):
                st.session_state.last_search = None
                st.rerun()


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()
st.caption("KE Workbench v1.0 | Internal Use Only")

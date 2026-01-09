"""
Cross-Border Compliance Navigator.

Multi-jurisdiction compliance analysis with conflict detection and pathway synthesis.

Run from repo root:
    streamlit run frontend/Home.py
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import asyncio
from datetime import datetime

from backend.rule_service.app.services.jurisdiction.resolver import resolve_jurisdictions, get_equivalences
from backend.rule_service.app.services.jurisdiction.evaluator import evaluate_jurisdiction
from backend.rule_service.app.services.jurisdiction.conflicts import detect_conflicts
from backend.rule_service.app.services.jurisdiction.pathway import (
    synthesize_pathway,
    aggregate_obligations,
    estimate_timeline,
)
from backend.database_service.app.services.database import init_db_with_seed

# Initialize database
init_db_with_seed()

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Navigator",
    page_icon="🧭",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title("Cross-Border Compliance Navigator")

st.markdown("""
Navigate multi-jurisdiction compliance requirements for digital assets.
Select your scenario to get a comprehensive analysis of applicable regulations,
potential conflicts, and recommended compliance pathway.
""")

st.divider()

# -----------------------------------------------------------------------------
# Scenario Input
# -----------------------------------------------------------------------------

st.header("1. Define Your Scenario")

col1, col2 = st.columns(2)

with col1:
    issuer_jurisdiction = st.selectbox(
        "Issuer Jurisdiction",
        options=["EU", "UK", "US", "CH", "SG"],
        index=3,  # Default to CH
        help="Where is the issuer/operator based?",
    )

    instrument_type = st.selectbox(
        "Instrument Type",
        options=[
            "stablecoin",
            "crypto_asset",
            "tokenized_bond",
            "utility_token",
            "art",
            "emt",
        ],
        help="Type of digital asset or token",
    )

    activity = st.selectbox(
        "Activity",
        options=[
            "public_offer",
            "financial_promotion",
            "custody",
            "exchange",
            "transfer",
        ],
        help="Regulatory activity being performed",
    )

with col2:
    target_jurisdictions = st.multiselect(
        "Target Markets",
        options=["EU", "UK", "US", "CH", "SG"],
        default=["EU", "UK"],
        help="Markets where you intend to offer or promote",
    )

    investor_types = st.multiselect(
        "Investor Types",
        options=["retail", "professional", "institutional"],
        default=["professional"],
        help="Types of investors you're targeting",
    )

# Additional facts
with st.expander("Additional Scenario Details (Optional)"):
    col_facts1, col_facts2 = st.columns(2)

    with col_facts1:
        is_authorized = st.checkbox("Has existing authorization", value=False)
        is_credit_institution = st.checkbox("Is a credit institution", value=False)
        has_whitepaper = st.checkbox("Has prepared whitepaper", value=False)

    with col_facts2:
        is_fca_authorized = st.checkbox("FCA authorized (UK)", value=False)
        has_risk_warning = st.checkbox("Has risk warning", value=False)
        is_first_time_investor = st.checkbox("First-time investor scenario", value=False)

# Build facts dict
additional_facts = {}
if is_authorized:
    additional_facts["has_authorization"] = True
    additional_facts["issuer_type"] = "credit_institution" if is_credit_institution else "other"
if has_whitepaper:
    additional_facts["whitepaper_submitted"] = True
if is_fca_authorized:
    additional_facts["is_fca_authorized"] = True
if has_risk_warning:
    additional_facts["has_prescribed_risk_warning"] = True
    additional_facts["risk_warning_prominent"] = True
if is_first_time_investor:
    additional_facts["is_first_time_investor"] = True

# -----------------------------------------------------------------------------
# Run Analysis
# -----------------------------------------------------------------------------

st.divider()

if st.button("Analyze Compliance Requirements", type="primary", use_container_width=True):
    with st.spinner("Analyzing cross-border compliance requirements..."):
        # Step 1: Resolve jurisdictions
        applicable = resolve_jurisdictions(
            issuer=issuer_jurisdiction,
            targets=target_jurisdictions,
            instrument_type=instrument_type,
        )

        # Step 2: Get equivalences
        equivalences = get_equivalences(
            from_jurisdiction=issuer_jurisdiction,
            to_jurisdictions=target_jurisdictions,
        )

        # Step 3: Evaluate each jurisdiction
        async def run_evaluations():
            tasks = [
                evaluate_jurisdiction(
                    jurisdiction=j.jurisdiction.value,
                    regime_id=j.regime_id,
                    facts={
                        **additional_facts,
                        "instrument_type": instrument_type,
                        "activity": activity,
                        "investor_types": investor_types,
                        "target_jurisdiction": j.jurisdiction.value,
                        "jurisdiction": j.jurisdiction.value,
                    },
                )
                for j in applicable
            ]
            return await asyncio.gather(*tasks)

        jurisdiction_results = asyncio.run(run_evaluations())

        # Add roles
        for i, result in enumerate(jurisdiction_results):
            result["role"] = applicable[i].role.value

        # Step 4: Detect conflicts
        conflicts = detect_conflicts(jurisdiction_results)

        # Step 5: Synthesize pathway
        pathway = synthesize_pathway(jurisdiction_results, conflicts, equivalences)

        # Step 6: Aggregate obligations
        cumulative_obligations = aggregate_obligations(jurisdiction_results)

        # Store in session state for display
        st.session_state["navigate_results"] = {
            "applicable": applicable,
            "equivalences": equivalences,
            "jurisdiction_results": jurisdiction_results,
            "conflicts": conflicts,
            "pathway": pathway,
            "obligations": cumulative_obligations,
            "timeline": estimate_timeline(pathway),
        }

# -----------------------------------------------------------------------------
# Display Results
# -----------------------------------------------------------------------------

if "navigate_results" in st.session_state:
    results = st.session_state["navigate_results"]

    st.header("2. Analysis Results")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Jurisdictions",
            len(results["applicable"]),
            help="Number of applicable jurisdictions",
        )

    with col2:
        total_obligations = len(results["obligations"])
        st.metric(
            "Total Obligations",
            total_obligations,
            help="Cumulative obligations across all jurisdictions",
        )

    with col3:
        conflict_count = len(results["conflicts"])
        st.metric(
            "Conflicts Detected",
            conflict_count,
            delta=f"{conflict_count} issues" if conflict_count > 0 else None,
            delta_color="inverse" if conflict_count > 0 else "off",
        )

    with col4:
        st.metric(
            "Estimated Timeline",
            results["timeline"],
            help="Estimated time to full compliance",
        )

    st.divider()

    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs([
        "Jurisdictions",
        "Conflicts",
        "Pathway",
        "Audit Trail",
    ])

    with tab1:
        st.subheader("Jurisdiction Analysis")

        for jr in results["jurisdiction_results"]:
            with st.expander(
                f"**{jr['jurisdiction']}** - {jr['regime_id']} ({jr.get('role', 'unknown')})"
            ):
                # Status badge
                status = jr.get("status", "unknown")
                if status == "compliant":
                    st.success(f"Status: {status.upper()}")
                elif status == "blocked":
                    st.error(f"Status: {status.upper()}")
                elif status == "requires_action":
                    st.warning(f"Status: {status.upper()}")
                else:
                    st.info(f"Status: {status}")

                st.write(f"**Rules Evaluated:** {jr.get('rules_evaluated', 0)}")
                st.write(f"**Applicable Rules:** {jr.get('applicable_rules', 0)}")

                if jr.get("decisions"):
                    st.write("**Decisions:**")
                    for dec in jr["decisions"]:
                        st.write(f"- `{dec['rule_id']}`: {dec['decision']}")

                if jr.get("obligations"):
                    st.write("**Obligations:**")
                    for obl in jr["obligations"]:
                        st.write(f"- **{obl['id']}**: {obl.get('description', 'N/A')}")

    with tab2:
        st.subheader("Cross-Jurisdiction Conflicts")

        if not results["conflicts"]:
            st.success("No conflicts detected between jurisdictions")
        else:
            for conflict in results["conflicts"]:
                severity = conflict.get("severity", "info")
                if severity == "blocking":
                    st.error(f"**BLOCKING**: {conflict.get('description', 'Unknown conflict')}")
                elif severity == "warning":
                    st.warning(f"**WARNING**: {conflict.get('description', 'Unknown conflict')}")
                else:
                    st.info(f"**INFO**: {conflict.get('description', 'Unknown conflict')}")

                st.write(f"- Type: `{conflict.get('type')}`")
                st.write(f"- Jurisdictions: {', '.join(conflict.get('jurisdictions', []))}")
                if conflict.get("resolution_strategy"):
                    st.write(f"- Resolution: {conflict.get('resolution_note', conflict.get('resolution_strategy'))}")

    with tab3:
        st.subheader("Compliance Pathway")

        if not results["pathway"]:
            st.info("No compliance steps required")
        else:
            for step in results["pathway"]:
                status_icon = "✅" if step.get("status") == "waived" else "⏳"
                status_text = "WAIVED" if step.get("status") == "waived" else "PENDING"

                st.markdown(f"""
                **Step {step['step_id']}** {status_icon} `{status_text}`

                - **Jurisdiction:** {step.get('jurisdiction', 'N/A')}
                - **Regime:** {step.get('regime', 'N/A')}
                - **Action:** {step.get('action', step.get('obligation_id', 'N/A'))}
                - **Timeline:** {step.get('timeline', {}).get('min_days', '?')}-{step.get('timeline', {}).get('max_days', '?')} days
                """)

                if step.get("waiver_reason"):
                    st.caption(f"Waiver: {step['waiver_reason']}")

                if step.get("prerequisites"):
                    st.caption(f"Prerequisites: Steps {step['prerequisites']}")

                st.divider()

    with tab4:
        st.subheader("Cumulative Obligations")

        if not results["obligations"]:
            st.info("No obligations identified")
        else:
            # Group by jurisdiction
            obls_by_jurisdiction = {}
            for obl in results["obligations"]:
                j = obl.get("jurisdiction", "Unknown")
                if j not in obls_by_jurisdiction:
                    obls_by_jurisdiction[j] = []
                obls_by_jurisdiction[j].append(obl)

            for jurisdiction, obls in obls_by_jurisdiction.items():
                st.write(f"**{jurisdiction}** ({len(obls)} obligations)")
                for obl in obls:
                    st.write(f"- `{obl['id']}`: {obl.get('description', 'N/A')}")

        st.divider()

        # Equivalences
        st.subheader("Equivalence Determinations")

        if not results["equivalences"]:
            st.info("No equivalence determinations found")
        else:
            for eq in results["equivalences"]:
                st.write(f"**{eq['from']} → {eq['to']}** ({eq['scope']})")
                st.write(f"- Status: `{eq['status']}`")
                if eq.get("notes"):
                    st.write(f"- Notes: {eq['notes']}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()

st.caption("""
**Disclaimer:** This is a research/demo tool. The compliance analysis is illustrative
and should not be relied upon for actual regulatory decisions. Always consult qualified
legal counsel for compliance matters.
""")

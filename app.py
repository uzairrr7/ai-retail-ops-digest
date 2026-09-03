"""
app.py
------
Streamlit demo for the AI-Powered Retail Operations Digest project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "source_code"))

import streamlit as st
import pandas as pd

from data_audit import load_raw, audit_report
from cleaning import clean_all, build_weekly_store_table

st.set_page_config(page_title="AI Retail Ops Digest", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Retail Operations Digest")
st.caption(
    "A personal project exploring how to combine data engineering with LLM-generated "
    "reports while preventing AI hallucination through automated fact-checking."
)

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
This app turns raw retail data (transactions, staffing shifts, returns, stores) into a
weekly, per-store metrics table, then uses an LLM to generate a short digest — with
**every numeric claim the LLM makes checked against the real computed data** before
it's shown here.

**Pipeline:** Raw CSVs → Data Audit → Clean + Join → Weekly Metrics → AI Digest
(structured claims, verified) → Programmatic Claim Verification

[View the full source code and notebook on GitHub](https://github.com/uzairrr7/ai-retail-ops-digest)
"""
    )


@st.cache_data
def get_data():
    stores, transactions, shifts, returns = load_raw()
    report = audit_report(stores, transactions, shifts, returns)
    stores_c, tx_c, shifts_c, returns_c = clean_all()
    weekly = build_weekly_store_table(stores_c, tx_c, shifts_c, returns_c)
    return stores, report, weekly


stores, audit, weekly = get_data()

tab1, tab2, tab3 = st.tabs(["🔍 Data Audit", "📈 Weekly Metrics", "🤖 AI Digest"])

with tab1:
    st.subheader("Data Quality Audit")
    st.write("Automated checks run against the raw data before any cleaning.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Invalid store references (transactions)", audit["invalid_store_refs"]["transactions"]["count"])
        st.metric("Duplicate transaction rows", audit["duplicates"]["transactions_full_row"])
    with col2:
        st.metric("Invalid store references (shifts)", audit["invalid_store_refs"]["shifts"]["count"])
        missing_combos = len(audit["missing_store_week_combinations"])
        st.metric("Missing store/week combinations", missing_combos)

    if audit["missing_store_week_combinations"]:
        st.warning("**Missing data found:**")
        for store, wk, note in audit["missing_store_week_combinations"]:
            st.write(f"- Store `{store}`, Week {wk}: {note}")

    st.info(
        "Store `S09` (referenced in the raw data) does not exist in `stores.csv` and is "
        "excluded from all analysis below, rather than silently included."
    )

with tab2:
    st.subheader("Weekly Per-Store Metrics")
    store_ids = sorted(stores["store_id"])
    c1, c2 = st.columns(2)
    with c1:
        sel_store = st.selectbox("Store", store_ids)
    with c2:
        sel_week = st.selectbox("Week", list(range(1, 9)), index=4)

    row = weekly[(weekly["store_id"] == sel_store) & (weekly["week"] == sel_week)]
    if row.empty:
        st.error("No data for this store/week.")
    else:
        r = row.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Weekly Revenue", f"${r['weekly_revenue']:,.2f}")
        m2.metric("Transactions", f"{r['weekly_transaction_count']:.0f}")
        m3.metric("Return Rate", f"{r['weekly_return_rate']*100:.2f}%")
        hrs = r["weekly_staffing_hours"]
        m4.metric("Staffing Hours", "N/A (missing)" if pd.isna(hrs) else f"{hrs:.1f}")
        sph = r["sales_per_staffed_hour"]
        m5.metric("Sales / Staffed Hour", "N/A" if pd.isna(sph) else f"${sph:.2f}")

        st.dataframe(weekly[weekly["store_id"] == sel_store].round(2), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("AI-Generated Digest (Structured + Verified)")

    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    if not has_key:
        st.warning(
            "No `GEMINI_API_KEY` is configured for this deployment, so live AI generation "
            "is disabled here. The audit and metrics tabs above run fully without an API "
            "key. See the [GitHub repo](https://github.com/uzairrr7/ai-retail-ops-digest) "
            "to run the full pipeline (including AI generation) locally with your own key."
        )
    else:
        store_ids = sorted(stores["store_id"])
        c1, c2 = st.columns(2)
        with c1:
            sel_store2 = st.selectbox("Store", store_ids, key="ai_store")
        with c2:
            sel_week2 = st.selectbox("Week", list(range(2, 9)), key="ai_week")

        if st.button("Generate verified digest"):
            with st.spinner("Generating and verifying..."):
                try:
                    from digest_approach2 import generate_claims_approach2, build_verified_digest
                    claims, current, previous = generate_claims_approach2(weekly, sel_store2, sel_week2)
                    digest_text, results = build_verified_digest(claims, weekly, sel_store2, sel_week2)
                    st.markdown(digest_text)
                    st.divider()
                    st.write("**Verification detail:**")
                    for r in results:
                        icon = "✅" if r["status"] == "PASS" else "❌"
                        st.write(f"{icon} {r['status']} — claimed: {r['claimed_value']}, actual: {r['actual_value']}")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

st.divider()
st.caption("Built by Uzair · [Source on GitHub](https://github.com/uzairrr7/ai-retail-ops-digest)")
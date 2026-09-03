"""
digest_approach1.py
--------------------
Part 4, Approach 1 - Prompt-Based Grounding.

We give the LLM the calculated metrics for one store/week (plus the
previous week for comparison) and instruct it to write a short digest
using ONLY those numbers. There is no structured intermediate step and
no automatic verification built into the generation itself - the
output is free text that we verify afterwards using verifier.py.
"""

from llm_client import call_llm

SYSTEM_PROMPT = (
    "You are a retail operations analyst writing a short weekly digest "
    "for a Regional Multi-Store Lead. You must use ONLY the numbers given "
    "to you below. Do not invent, estimate, or round to a nicer-sounding "
    "figure. Do not mention any store, week, or metric that is not in the "
    "data provided. If data for a metric is missing (shown as N/A), say so "
    "explicitly rather than guessing a value."
)


def build_metrics_block(current: dict, previous: dict | None) -> str:
    lines = [f"Store: {current['store_id']} (Region: {current['region']})  |  Week: {current['week']}"]
    lines.append(f"- Weekly Revenue: ${current['weekly_revenue']:.2f}")
    lines.append(f"- Weekly Transaction Count: {current['weekly_transaction_count']:.0f}")
    lines.append(f"- Weekly Return Rate: {current['weekly_return_rate']*100:.2f}%")
    hrs = current['weekly_staffing_hours']
    lines.append(f"- Weekly Staffing Hours: {'N/A (missing staffing data)' if pd_isna(hrs) else f'{hrs:.1f}'}")
    sph = current['sales_per_staffed_hour']
    lines.append(f"- Sales per Staffed Hour: {'N/A (cannot compute - staffing hours missing)' if pd_isna(sph) else f'${sph:.2f}'}")
    lines.append(f"- Online Sales Share: {current['online_sales_share_pct']:.1f}%")
    lines.append(f"- Revenue per Sq Ft: ${current['revenue_per_sqft']:.2f}")

    if previous is not None:
        lines.append(f"\nPrevious week ({previous['week']}) revenue for comparison: ${previous['weekly_revenue']:.2f}")
        wow = current.get('revenue_wow_change_pct')
        if wow is not None and not pd_isna(wow):
            lines.append(f"Week-over-week revenue change (pre-calculated - use this exact figure if you mention it): {wow:.2f}%")

    return "\n".join(lines)


def pd_isna(x):
    import pandas as pd
    return pd.isna(x)


def generate_digest_approach1(metrics_df, store_id: str, week: int) -> str:
    row = metrics_df[(metrics_df["store_id"] == store_id) & (metrics_df["week"] == week)]
    if row.empty:
        raise ValueError(f"No data for {store_id} / week {week}")
    current = row.iloc[0].to_dict()

    prev_row = metrics_df[(metrics_df["store_id"] == store_id) & (metrics_df["week"] == week - 1)]
    previous = prev_row.iloc[0].to_dict() if not prev_row.empty else None

    metrics_block = build_metrics_block(current, previous)

    prompt = (
        f"Here are this store's calculated weekly metrics:\n\n{metrics_block}\n\n"
        "Write a short weekly operations digest (4-6 sentences) covering:\n"
        "1. Key observations\n2. Important changes\n3. Areas requiring attention\n"
        "4. A short recommended focus\n\n"
        "Use only the numbers above. If you state a percentage change, use the "
        "pre-calculated week-over-week figure given above rather than computing your own."
    )

    digest_text = call_llm(prompt, system=SYSTEM_PROMPT)
    return digest_text, current, previous

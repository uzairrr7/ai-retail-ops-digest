"""
verifier.py
-----------
Part 5 of the assignment: a function that checks whether a numerical
claim made by the LLM is actually supported by the computed metrics.

Design choice: instead of trying to regex-parse arbitrary free text
(fragile, easy to fool), the verifier takes a STRUCTURED claim:

    {
        "store_id": "S01",
        "week": 5,
        "metric": "weekly_revenue",       # which column in the metrics table
        "claim_type": "absolute",          # "absolute" or "change_percent"
        "value": 14315.48
    }

For "change_percent" claims, the verifier recomputes the week-over-week
percentage change itself from the metrics table (current vs previous
week for that store) and compares it to the claimed value.

This structured shape is what lets Approach 2 (Structured Generation +
Verification, Part 4) work: the LLM is asked to output claims in this
exact JSON shape, and every single one is run through this function
before being allowed into the final digest text.

For Approach 1 (free-text digest), we still verify it, but a lightweight
text-claim extractor (extract_claims_from_text) turns simple sentences
into this same structured shape first, using the metrics table as the
source of truth for which numbers "belong" to which store/week - this
is necessarily best-effort since free text is ambiguous, which is
itself one of the findings in Part 4's approach comparison.
"""

import re
import pandas as pd

ABS_TOLERANCE = 1.0          # currency values: allow $1 rounding slack
PCT_TOLERANCE = 0.5          # percentage points: allow 0.5pp slack


def _get_actual_value(metrics_df: pd.DataFrame, store_id: str, week: int, metric: str):
    row = metrics_df[(metrics_df["store_id"] == store_id) & (metrics_df["week"] == week)]
    if row.empty:
        return None
    if metric not in row.columns:
        return None
    val = row[metric].iloc[0]
    return None if pd.isna(val) else float(val)


def _get_actual_change_pct(metrics_df: pd.DataFrame, store_id: str, week: int, metric: str):
    """Recompute % change vs the previous week for `metric`, from scratch."""
    current = _get_actual_value(metrics_df, store_id, week, metric)
    previous = _get_actual_value(metrics_df, store_id, week - 1, metric)
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / previous * 100


def verify_claim(claim: dict, metrics_df: pd.DataFrame) -> dict:
    """
    Verify a single structured claim against the metrics table.

    Returns:
        {
            "status": "PASS" | "FAIL",
            "reason": str,
            "claimed_value": float,
            "actual_value": float | None,
        }
    """
    store_id = claim.get("store_id")
    week = claim.get("week")
    metric = claim.get("metric")
    claim_type = claim.get("claim_type", "absolute")
    claimed_value = claim.get("value")

    if store_id is None or week is None or metric is None or claimed_value is None:
        return {"status": "FAIL", "reason": "Malformed claim (missing required field)",
                "claimed_value": claimed_value, "actual_value": None}

    if claim_type == "absolute":
        actual = _get_actual_value(metrics_df, store_id, week, metric)
        if actual is None:
            return {"status": "FAIL", "reason": f"No data for {store_id}/week {week}/{metric}",
                    "claimed_value": claimed_value, "actual_value": None}
        if abs(actual - claimed_value) <= ABS_TOLERANCE:
            return {"status": "PASS", "reason": "Matches computed value within tolerance",
                    "claimed_value": claimed_value, "actual_value": actual}
        return {"status": "FAIL", "reason": "Unsupported numerical claim - does not match computed value",
                "claimed_value": claimed_value, "actual_value": actual}

    elif claim_type == "change_percent":
        actual = _get_actual_change_pct(metrics_df, store_id, week, metric)
        if actual is None:
            return {"status": "FAIL", "reason": f"Cannot compute change for {store_id}/week {week}/{metric}",
                    "claimed_value": claimed_value, "actual_value": None}
        if abs(actual - claimed_value) <= PCT_TOLERANCE:
            return {"status": "PASS", "reason": "Matches computed % change within tolerance",
                    "claimed_value": claimed_value, "actual_value": actual}
        return {"status": "FAIL", "reason": "Unsupported numerical claim - % change does not match",
                "claimed_value": claimed_value, "actual_value": actual}

    return {"status": "FAIL", "reason": f"Unknown claim_type '{claim_type}'",
            "claimed_value": claimed_value, "actual_value": None}


def verify_claims(claims: list, metrics_df: pd.DataFrame) -> list:
    """Verify a list of structured claims. Returns list of results (same order)."""
    return [dict(claim=c, **verify_claim(c, metrics_df)) for c in claims]


# ---------------------------------------------------------------------
# Best-effort free-text claim extractor, used for Approach 1.
# Looks for simple "$X" and "increased/decreased by X%" patterns.
# This is intentionally simple - Part 4's comparison notes that this
# is a real limitation of the prompt-only approach vs. Approach 2.
# ---------------------------------------------------------------------
_DOLLAR_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)")
_PCT_CHANGE_RE = re.compile(
    r"(increased|decreased|rose|fell|grew|dropped)\s+(?:by\s+)?([\d.]+)\s*%", re.IGNORECASE
)


def extract_claims_from_text(text: str, store_id: str, week: int, metric_for_pct="weekly_revenue"):
    """
    Best-effort extraction of structured claims from free text digest,
    for a KNOWN store_id/week context (the digest is written per store/week,
    so we know which store/week the sentence is talking about - test case 5
    covers what happens when that assumption is deliberately violated).
    """
    claims = []
    for m in _DOLLAR_RE.finditer(text):
        value = float(m.group(1).replace(",", ""))
        claims.append({
            "store_id": store_id, "week": week, "metric": "weekly_revenue",
            "claim_type": "absolute", "value": value,
        })
    for m in _PCT_CHANGE_RE.finditer(text):
        direction, pct = m.group(1).lower(), float(m.group(2))
        signed = pct if direction in ("increased", "rose", "grew") else -pct
        claims.append({
            "store_id": store_id, "week": week, "metric": metric_for_pct,
            "claim_type": "change_percent", "value": signed,
        })
    return claims

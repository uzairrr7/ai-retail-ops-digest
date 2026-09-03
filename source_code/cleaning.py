"""
cleaning.py
-----------
Applies the cleaning decisions documented in the audit (see
data_audit.py and Technical_Summary) and builds the weekly, per-store
dataset used for metrics.

Cleaning decisions applied here:
  1. Drop transactions/shifts referencing store S09 (not a real store).
  2. Drop exact-duplicate transaction rows (keep first occurrence).
  3. Treat blank promo_code as "no promo used" (not missing data).
  4. Leave the S04/week-6 staffing gap as a true gap (NaN hours) rather
     than inventing a value - it is surfaced explicitly in the metrics
     and digest instead of being silently patched.
"""

import pandas as pd
from data_audit import load_raw

WEEK1_START = pd.Timestamp("2025-04-07")


def week_number(dates: pd.Series) -> pd.Series:
    return ((pd.to_datetime(dates) - WEEK1_START).dt.days // 7) + 1


def clean_all():
    stores, transactions, shifts, returns = load_raw()
    valid_store_ids = set(stores["store_id"])

    # 1. Drop invalid store references
    transactions = transactions[transactions["store_id"].isin(valid_store_ids)].copy()
    shifts = shifts[shifts["store_id"].isin(valid_store_ids)].copy()
    returns = returns[returns["store_id"].isin(valid_store_ids)].copy()

    # 2. Drop exact duplicate transactions
    transactions = transactions.drop_duplicates(subset=["transaction_id"], keep="first")

    # 3. promo_code: blank -> "NONE" flag (not a null to impute)
    transactions["promo_code"] = transactions["promo_code"].fillna("NONE")

    # Parse dates / add week numbers
    transactions["week"] = week_number(transactions["timestamp"])
    shifts["week"] = week_number(shifts["date"])
    returns["week"] = week_number(returns["date"])

    return stores, transactions, shifts, returns


def build_weekly_store_table(stores, transactions, shifts, returns):
    """
    Produces one row per (store_id, week) with the 5 required metrics
    plus 3 extra metrics chosen for the Regional Multi-Store Lead
    perspective (see Technical Summary for justification).
    """
    weeks = range(1, 9)
    store_ids = sorted(stores["store_id"])

    # Base skeleton: every store x every week (so gaps are visible, not silently dropped)
    skeleton = pd.MultiIndex.from_product([store_ids, weeks], names=["store_id", "week"]).to_frame(index=False)

    rev = transactions.groupby(["store_id", "week"])["amount"].sum().rename("weekly_revenue")
    txn_count = transactions.groupby(["store_id", "week"])["transaction_id"].count().rename("weekly_transaction_count")
    online_count = (
        transactions[transactions["channel"] == "online"]
        .groupby(["store_id", "week"])["transaction_id"].count()
        .rename("online_transaction_count")
    )
    ret_amt = returns.groupby(["store_id", "week"])["amount"].sum().rename("weekly_return_amount")
    hours = shifts.groupby(["store_id", "week"])["hours_worked"].sum().rename("weekly_staffing_hours")
    distinct_staff = shifts.groupby(["store_id", "week"])["employee_id"].nunique().rename("distinct_staff_count")

    df = skeleton.merge(rev, on=["store_id", "week"], how="left")
    df = df.merge(txn_count, on=["store_id", "week"], how="left")
    df = df.merge(online_count, on=["store_id", "week"], how="left")
    df = df.merge(ret_amt, on=["store_id", "week"], how="left")
    df = df.merge(hours, on=["store_id", "week"], how="left")
    df = df.merge(distinct_staff, on=["store_id", "week"], how="left")

    # Weeks/stores with genuinely zero activity vs missing data:
    # revenue/txn/returns default to 0 if absent (no sales recorded that week is a real 0)
    for col in ["weekly_revenue", "weekly_transaction_count", "online_transaction_count", "weekly_return_amount"]:
        df[col] = df[col].fillna(0)

    # Staffing hours: DO NOT fill with 0. A store/week with no shift
    # records is a data gap (see S04/week6), not a store that used 0
    # staff hours. We keep it as NaN so downstream metrics surface it.
    # (distinct_staff_count follows the same rule.)

    # --- 5 required metrics ---
    df["weekly_return_rate"] = df.apply(
        lambda r: (r["weekly_return_amount"] / r["weekly_revenue"]) if r["weekly_revenue"] > 0 else 0.0,
        axis=1,
    )
    df["sales_per_staffed_hour"] = df.apply(
        lambda r: (r["weekly_revenue"] / r["weekly_staffing_hours"])
        if pd.notna(r["weekly_staffing_hours"]) and r["weekly_staffing_hours"] > 0
        else None,
        axis=1,
    )

    # --- Extra metrics for Regional Multi-Store Lead perspective ---
    # a) week-over-week revenue change (%) per store
    df = df.sort_values(["store_id", "week"])
    df["revenue_wow_change_pct"] = df.groupby("store_id")["weekly_revenue"].pct_change() * 100

    # b) online sales share (%) - regional channel mix comparison
    df["online_sales_share_pct"] = df.apply(
        lambda r: (r["online_transaction_count"] / r["weekly_transaction_count"] * 100)
        if r["weekly_transaction_count"] > 0 else 0.0,
        axis=1,
    )

    # c) revenue per sq ft - lets us compare stores of very different sizes fairly
    sqft_map = stores.set_index("store_id")["size_sqft"].to_dict()
    df["revenue_per_sqft"] = df.apply(lambda r: r["weekly_revenue"] / sqft_map[r["store_id"]], axis=1)

    # attach region for convenience
    region_map = stores.set_index("store_id")["region"].to_dict()
    df["region"] = df["store_id"].map(region_map)

    cols = [
        "store_id", "region", "week",
        "weekly_revenue", "weekly_transaction_count", "weekly_return_rate",
        "weekly_staffing_hours", "sales_per_staffed_hour",
        "revenue_wow_change_pct", "online_sales_share_pct", "revenue_per_sqft",
    ]
    return df[cols].reset_index(drop=True)


if __name__ == "__main__":
    stores, transactions, shifts, returns = clean_all()
    weekly = build_weekly_store_table(stores, transactions, shifts, returns)
    pd.set_option("display.width", 160)
    print(weekly.round(2).to_string())
    print(f"\nRows with missing sales_per_staffed_hour (staffing data gap):")
    print(weekly[weekly["sales_per_staffed_hour"].isnull()][["store_id", "week"]])

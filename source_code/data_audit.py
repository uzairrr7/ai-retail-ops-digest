"""
data_audit.py
-------------
Loads the four raw CSV files and runs the data-quality audit described
in Part 1 of the assignment. Every issue found here is documented in
Problem -> Evidence -> Decision -> Reason format in the notebook /
Technical Summary, not just fixed silently.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw():
    """Load the four raw CSVs exactly as provided (no cleaning yet)."""
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    shifts = pd.read_csv(DATA_DIR / "staffing_shifts.csv")
    returns = pd.read_csv(DATA_DIR / "returns.csv")
    return stores, transactions, shifts, returns


def audit_report(stores, transactions, shifts, returns):
    """
    Runs a set of checks and returns a dict summarising what was found.
    This does NOT modify the data - it only reports on it. Cleaning
    decisions based on these findings live in cleaning.py.
    """
    report = {}

    valid_store_ids = set(stores["store_id"])

    # --- Referential integrity: store_id must exist in stores.csv ---
    tx_bad_store = transactions[~transactions["store_id"].isin(valid_store_ids)]
    shifts_bad_store = shifts[~shifts["store_id"].isin(valid_store_ids)]
    returns_bad_store = returns[~returns["store_id"].isin(valid_store_ids)]

    report["invalid_store_refs"] = {
        "transactions": {
            "count": len(tx_bad_store),
            "unknown_store_ids": sorted(tx_bad_store["store_id"].unique().tolist()),
        },
        "shifts": {
            "count": len(shifts_bad_store),
            "unknown_store_ids": sorted(shifts_bad_store["store_id"].unique().tolist()),
        },
        "returns": {
            "count": len(returns_bad_store),
            "unknown_store_ids": sorted(returns_bad_store["store_id"].unique().tolist()),
        },
    }

    # --- Duplicate records ---
    report["duplicates"] = {
        "transactions_full_row": int(transactions.duplicated().sum()),
        "transactions_by_id": int(transactions.duplicated(subset=["transaction_id"]).sum()),
        "shifts_full_row": int(shifts.duplicated().sum()),
        "returns_full_row": int(returns.duplicated().sum()),
    }

    # --- Missing values ---
    report["nulls"] = {
        "transactions": transactions.isnull().sum().to_dict(),
        "shifts": shifts.isnull().sum().to_dict(),
        "returns": returns.isnull().sum().to_dict(),
    }

    # --- Invalid / unparseable dates ---
    tx_ts = pd.to_datetime(transactions["timestamp"], errors="coerce")
    shift_dates = pd.to_datetime(shifts["date"], errors="coerce")
    return_dates = pd.to_datetime(returns["date"], errors="coerce")

    report["invalid_dates"] = {
        "transactions": int(tx_ts.isnull().sum()),
        "shifts": int(shift_dates.isnull().sum()),
        "returns": int(return_dates.isnull().sum()),
    }

    # --- Value range sanity checks ---
    report["value_ranges"] = {
        "transactions_amount_min_max": (float(transactions["amount"].min()), float(transactions["amount"].max())),
        "transactions_nonpositive_amount": int((transactions["amount"] <= 0).sum()),
        "shifts_hours_min_max": (float(shifts["hours_worked"].min()), float(shifts["hours_worked"].max())),
        "shifts_nonpositive_hours": int((shifts["hours_worked"] <= 0).sum()),
    }

    # --- Missing store/week combinations (using only valid stores) ---
    start = pd.Timestamp("2025-04-07")  # Week 1 start (first date seen in data)

    tx_valid = transactions[transactions["store_id"].isin(valid_store_ids)].copy()
    tx_valid["week"] = ((pd.to_datetime(tx_valid["timestamp"]) - start).dt.days // 7) + 1

    shifts_valid = shifts[shifts["store_id"].isin(valid_store_ids)].copy()
    shifts_valid["week"] = ((pd.to_datetime(shifts_valid["date"]) - start).dt.days // 7) + 1

    all_weeks = range(1, 9)
    missing_combos = []
    for store in sorted(valid_store_ids):
        weeks_with_shifts = set(shifts_valid[shifts_valid["store_id"] == store]["week"])
        for wk in all_weeks:
            if wk not in weeks_with_shifts:
                missing_combos.append((store, wk, "no staffing_shifts records"))

    report["missing_store_week_combinations"] = missing_combos

    return report


def print_audit_summary(report):
    """Human-readable console summary of the audit report."""
    print("=== INVALID STORE REFERENCES (not in stores.csv) ===")
    for src, d in report["invalid_store_refs"].items():
        print(f"  {src}: {d['count']} records referencing {d['unknown_store_ids']}")

    print("\n=== DUPLICATES ===")
    for k, v in report["duplicates"].items():
        print(f"  {k}: {v}")

    print("\n=== MISSING VALUES ===")
    for src, nulls in report["nulls"].items():
        nonzero = {k: v for k, v in nulls.items() if v > 0}
        print(f"  {src}: {nonzero if nonzero else 'none'}")

    print("\n=== INVALID / UNPARSEABLE DATES ===")
    for src, v in report["invalid_dates"].items():
        print(f"  {src}: {v}")

    print("\n=== VALUE RANGE CHECKS ===")
    for k, v in report["value_ranges"].items():
        print(f"  {k}: {v}")

    print("\n=== MISSING STORE/WEEK COMBINATIONS (staffing) ===")
    if report["missing_store_week_combinations"]:
        for store, wk, note in report["missing_store_week_combinations"]:
            print(f"  {store} / week {wk}: {note}")
    else:
        print("  none found")


if __name__ == "__main__":
    stores, transactions, shifts, returns = load_raw()
    report = audit_report(stores, transactions, shifts, returns)
    print_audit_summary(report)

"""
test_verifier.py
-----------------
Part 10 of the assignment: the 5 required verification test cases.
Uses the exact example numbers given in the brief (Revenue = $100,000,
Previous Week Revenue = $90,000) via a small fixture table, plus one
test drawn from our real project data for Test 5.

Run with:  python3 test_verifier.py
"""

import pandas as pd
from verifier import verify_claim

# Fixture matching the brief's worked example exactly:
# Store A: week 2 revenue = 100,000 ; week 1 revenue = 90,000
# Store B: week 2 revenue = 40,000   (used only for Test 5 - wrong store)
fixture = pd.DataFrame([
    {"store_id": "A", "week": 1, "weekly_revenue": 90000},
    {"store_id": "A", "week": 2, "weekly_revenue": 100000},
    {"store_id": "B", "week": 1, "weekly_revenue": 35000},
    {"store_id": "B", "week": 2, "weekly_revenue": 40000},
])


def run_test(name, claim, expected_status):
    result = verify_claim(claim, fixture)
    status = result["status"]
    ok = "OK " if status == expected_status else "MISMATCH!"
    print(f"[{ok}] {name}: expected={expected_status} got={status} | "
          f"claimed={result['claimed_value']} actual={result['actual_value']} | {result['reason']}")
    assert status == expected_status, f"{name} failed: expected {expected_status}, got {status}"


if __name__ == "__main__":
    # Test 1 - Correct number -> PASS
    run_test(
        "Test 1: Correct number (Revenue = $100,000)",
        {"store_id": "A", "week": 2, "metric": "weekly_revenue", "claim_type": "absolute", "value": 100000},
        "PASS",
    )

    # Test 2 - Incorrect number -> FAIL
    run_test(
        "Test 2: Incorrect number (Revenue = $150,000)",
        {"store_id": "A", "week": 2, "metric": "weekly_revenue", "claim_type": "absolute", "value": 150000},
        "FAIL",
    )

    # Test 3 - Correct percentage -> PASS
    # (100000 - 90000) / 90000 * 100 = 11.11%
    run_test(
        "Test 3: Correct percentage (+11.1%)",
        {"store_id": "A", "week": 2, "metric": "weekly_revenue", "claim_type": "change_percent", "value": 11.1},
        "PASS",
    )

    # Test 4 - Incorrect percentage -> FAIL
    run_test(
        "Test 4: Incorrect percentage (+25%)",
        {"store_id": "A", "week": 2, "metric": "weekly_revenue", "claim_type": "change_percent", "value": 25},
        "FAIL",
    )

    # Test 5 - Wrong metric/store/week: claim uses Store B's actual
    # revenue value (40,000) but is asserted while talking about Store A
    # -> FAIL, because Store A's real week-2 revenue is 100,000, not 40,000
    run_test(
        "Test 5: Wrong store (Store B's value claimed for Store A)",
        {"store_id": "A", "week": 2, "metric": "weekly_revenue", "claim_type": "absolute", "value": 40000},
        "FAIL",
    )

    print("\nAll 5 required verification tests passed.")

import pytest

from app.budget import BudgetExceeded, BudgetTracker


def test_budget_allows_until_limit(tmp_path):
    bt = BudgetTracker(str(tmp_path / "budget.json"), daily_usd_limit=0.0, daily_calls_limit=3)

    bt.check_and_record()
    bt.check_and_record()
    bt.check_and_record()

    with pytest.raises(BudgetExceeded):
        bt.check_and_record()
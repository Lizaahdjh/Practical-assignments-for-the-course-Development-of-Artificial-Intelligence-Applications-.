import json
from datetime import datetime, timezone, date
from pathlib import Path

class BudgetTracker:
    """Відстежує денні витрати на API. Записує в JSON-файл."""
    
    def __init__(self, budget_file: str, daily_usd_limit: float, daily_calls_limit: int):
        self.budget_file = Path(budget_file)
        self.daily_usd_limit = daily_usd_limit
        self.daily_calls_limit = daily_calls_limit
        self._ensure_file()

    def _ensure_file(self):
        if not self.budget_file.exists():
            self.budget_file.parent.mkdir(parents=True, exist_ok=True)
            self._save({"date": str(date.today()), "calls": 0, "usd": 0.0})

    def _load(self) -> dict:
        try:
            data = json.loads(self.budget_file.read_text())
            if data.get("date") != str(date.today()):
                # Новий день — скидаємо лічильники
                return {"date": str(date.today()), "calls": 0, "usd": 0.0}
            return data
        except (json.JSONDecodeError, FileNotFoundError):
            return {"date": str(date.today()), "calls": 0, "usd": 0.0}

    def _save(self, data: dict) -> None:
        self.budget_file.write_text(json.dumps(data, indent=2))

    def check_and_record(self, estimated_cost_usd: float = 0.0) -> None:
        """Перевіряє ліміти і записує виклик. Кидає BudgetExceeded, якщо перевищено."""
        data = self._load()
        
        if self.daily_calls_limit > 0 and data["calls"] >= self.daily_calls_limit:
            raise BudgetExceeded(f"Досягнуто денного ліміту викликів: {self.daily_calls_limit}")
        
        if self.daily_usd_limit > 0.0 and (data["usd"] + estimated_cost_usd) > self.daily_usd_limit:
            raise BudgetExceeded(f"Досягнуто денного ліміту витрат: ${self.daily_usd_limit}")
        
        data["calls"] += 1
        data["usd"] += estimated_cost_usd
        self._save(data)


class BudgetExceeded(Exception):
    pass
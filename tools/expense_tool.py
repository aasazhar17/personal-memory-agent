import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any

class ExpenseTool:
    def __init__(self, db, data_dir: str = None):
        self.db = db
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "expenses.json")
        self.expenses: List[Dict[str, Any]] = []
        self._load_expenses()

    def _load_expenses(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.expenses = json.load(f)
            except:
                self.expenses = []
        else:
            self.expenses = []
            self._save_expenses()

    def _save_expenses(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.expenses, f, indent=4)

    async def add_expense(self, description: str, amount: float, date_str: str, category: str = "General") -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._add_expense_sync, description, amount, date_str, category)

    def _add_expense_sync(self, description: str, amount: float, date_str: str, category: str) -> Dict[str, Any]:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except:
            date_str = datetime.now().strftime("%Y-%m-%d")
        expense_id = len(self.expenses) + 1
        record = {
            "id": expense_id,
            "description": description.strip(),
            "amount": float(amount),
            "date": date_str,
            "category": category.strip()
        }
        self.expenses.append(record)
        self._save_expenses()
        # Index in vector DB
        text = f"Expense: {description} cost Rs.{amount} on {date_str} (Category: {category})"
        self.db.add_texts([text], [{
            "type": "expense",
            "expense_id": expense_id,
            "description": description,
            "amount": amount,
            "date": date_str,
            "category": category
        }])
        return {"success": True, "expense": record}

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.db.similarity_search, query, k)
        return [r for r in results if r.get("type") == "expense"]

    async def get_expenses_after_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Date (YYYY-MM-DD) ke baad ke expenses return kare."""
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return []
        result = []
        for e in self.expenses:
            try:
                exp_date = datetime.strptime(e["date"], "%Y-%m-%d")
                if exp_date >= target:
                    result.append(e)
            except:
                continue
        return result

    async def get_all_expenses(self) -> List[Dict[str, Any]]:
        return self.expenses
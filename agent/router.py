import re
import asyncio
from typing import List, Dict, Any
import numpy as np
from embeddings.embed import EmbeddingGenerator

class AgentRouter:
    def __init__(self, embedder=None):
        self.embedder = embedder or EmbeddingGenerator()
        self.tool_names = ["pdf_tool", "expense_tool", "notes_tool", "calculator_tool", "profile_tool"]
        # Semantic descriptions for each tool
        self.tool_descriptions = {
            "pdf_tool": ["search PDF documents", "bills", "reports", "tickets", "invoices", "MRI", "medical"],
            "expense_tool": ["expenses", "spending", "cost", "amount", "budget", "paid", "purchase"],
            "notes_tool": ["notes", "memos", "reminders", "EMI", "schedule", "dates", "trip"],
            "calculator_tool": ["calculate", "math", "arithmetic", "sum", "add", "subtract", "multiply", "divide"],
            "profile_tool": ["my name", "my college", "my city", "my pet", "my food", "profile", "who am i"]
        }
        self._cached_embeddings = {}

    def _get_tool_embeddings(self):
        if not self._cached_embeddings:
            for tool, desc_list in self.tool_descriptions.items():
                self._cached_embeddings[tool] = self.embedder.get_embeddings(desc_list)
        return self._cached_embeddings

    async def route(self, query: str, api_key: str = None) -> List[str]:
        # Try LLM routing if API key provided
        if api_key:
            try:
                return await self._route_with_llm(query, api_key)
            except:
                pass
        # Fallback to local routing
        return self.route_local(query)

    async def _route_with_llm(self, query: str, api_key: str) -> List[str]:
        import google.generativeai as genai
        import json
        genai.configure(api_key=api_key)
        
        prompt = f"""
You are an intelligent query router. Analyze the user query and determine which tools from the list below are relevant to answer the query. You can select one or multiple tools.

Available tools:
1. "pdf_tool": Use when searching information inside uploaded PDF documents, invoices, tickets, medical reports, prescriptions, scans, bills.
2. "expense_tool": Use when the query asks about money spent, costs, budgets, expenses, purchases, or pricing.
3. "notes_tool": Use when the query asks about notes, personal reminders, memos, schedules, trips, or event dates.
4. "calculator_tool": Use when the query contains a mathematical calculation or formula to evaluate.
5. "profile_tool": Use when the query asks about user's personal profile (e.g. name, college, city, birthday, pet, favourite food, job).

Output ONLY a valid JSON list containing the names of the selected tools (e.g. ["expense_tool", "notes_tool"]). Do not explain your choice.

User query: "{query}"
JSON List:
"""
        try:
            loop = asyncio.get_running_loop()
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
            )
            data = json.loads(response.text.strip())
            if isinstance(data, list):
                valid_tools = [t for t in data if t in self.tool_names]
                if valid_tools:
                    return valid_tools
        except Exception as e:
            pass
        return self.route_local(query)

    def route_local(self, query: str) -> List[str]:
        q = query.lower()
        selected = set()

        # 1. Rule-based triggers (fast)
        if any(op in q for op in ["+", "-", "*", "/", "%"]) or re.search(r'\d+\s*(plus|minus|times|divided by)', q):
            selected.add("calculator_tool")

        # Profile keywords - expanded
        profile_keywords = [
            "what is my", "what's my", "what do you know about me", "tell me about myself",
            "my name", "my city", "my college", "my university", "my school",
            "who am i", "my profile", "do you know my", "my favourite", "my favorite",
            "my pet", "my birthday", "my age", "my job", "my hobby", "my mother", "my father"
        ]
        if any(kw in q for kw in profile_keywords) or re.search(r'\bmy\s+(\w+)\s+is\s+(.+)', q):
            selected.add("profile_tool")

        # PDF documents
        pdf_keywords = ["pdf", "document", "ticket", "invoice", "bill", "receipt", "report",
                        "pnr", "railway", "medical", "mri", "prescription", "scan", "statement", "electricity"]
        if any(kw in q for kw in pdf_keywords):
            selected.add("pdf_tool")

        # Expenses - but avoid document-only terms
        expense_keywords = ["spend", "spent", "expense", "expenditure", "budget", "paid", "purchase", "cost", "price"]
        doc_only = any(kw in q for kw in ["ticket", "pnr", "invoice", "receipt", "mri", "prescription", "scan"])
        if any(kw in q for kw in expense_keywords) and not doc_only:
            selected.add("expense_tool")

        # Notes
        notes_keywords = ["note", "memo", "emi", "due", "remind", "trip", "date", "schedule", "reminder"]
        if any(kw in q for kw in notes_keywords):
            selected.add("notes_tool")

        # 2. Semantic similarity (if rule-based didn't select anything or low confidence)
        if not selected:
            try:
                q_emb = np.array(self.embedder.get_embedding(query))
                if q_emb.size > 0:
                    tool_embs = self._get_tool_embeddings()
                    scores = {}
                    for tool, embs in tool_embs.items():
                        max_sim = 0.0
                        for emb in embs:
                            e = np.array(emb)
                            sim = np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e) + 1e-8)
                            max_sim = max(max_sim, sim)
                        scores[tool] = max_sim
                    # Add tool if similarity > threshold
                    threshold = 0.35
                    for tool, score in scores.items():
                        if score > threshold:
                            selected.add(tool)
                    # Fallback: if still empty, choose highest scoring
                    if not selected and scores:
                        best = max(scores, key=scores.get)
                        selected.add(best)
            except:
                pass

        # Final fallback - if still empty, default to notes_tool
        if not selected:
            selected.add("notes_tool")

        return list(selected)
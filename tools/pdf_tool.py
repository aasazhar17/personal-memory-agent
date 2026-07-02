import os
import asyncio
import re
from typing import List, Dict, Any

class PDFTool:
    def __init__(self, db, profile_tool=None, notes_tool=None, expense_tool=None):
        self.db = db
        self.profile_tool = profile_tool
        self.notes_tool = notes_tool
        self.expense_tool = expense_tool

    async def ingest_pdf(self, file_path: str, api_key: str = None) -> Dict[str, Any]:
        """PDF read karo, chunk karo, vectorstore mein daalo. Aur agar API key ho, toh auto-extract karke details save karo."""
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self._ingest_pdf_sync, file_path)
        if not res.get("success"):
            return res
            
        # Auto-extract structured data if api_key is available
        if api_key:
            try:
                await self._auto_extract_and_save(file_path, api_key)
                res["message"] += " Automatically extracted and saved records from document."
            except Exception as e:
                res["message"] += f" (Auto-extraction skipped/failed: {str(e)})"
                
        return res

    def _ingest_pdf_sync(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {"success": False, "message": f"File '{file_path}' not found."}
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            filename = os.path.basename(file_path)
            chunks = []
            metadatas = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                page_chunks = self._chunk_text(text, chunk_size=600, overlap=100)
                for chunk in page_chunks:
                    chunks.append(chunk)
                    metadatas.append({
                        "source": filename,
                        "page": i + 1,
                        "type": "pdf"
                    })
            if chunks:
                self.db.add_texts(chunks, metadatas)
                return {
                    "success": True,
                    "message": f"Indexed {len(chunks)} chunks from {filename}.",
                    "num_chunks": len(chunks)
                }
            else:
                return {"success": False, "message": f"No text found in {filename}."}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        text = text.replace('\n', ' ').strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
            if start >= len(text) or end == len(text):
                break
        return chunks

    async def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """Vector search karo, specific filename match karo agar query mein hai."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.db.similarity_search, query, k)
        
        # Filter only PDF results
        pdf_results = [r for r in results if r.get("type") == "pdf"]
        
        # Agar query mein exact filename hai toh filter karo
        pdf_match = re.search(r'\b([\w\-&]+\.pdf)\b', query, re.IGNORECASE)
        if pdf_match:
            filename = pdf_match.group(1).strip().lower()
            pdf_results = [r for r in pdf_results if r.get("source", "").strip().lower() == filename]
        
        # Duplicates hatao
        unique = []
        seen = set()
        for r in pdf_results:
            text_clean = r.get("text", "").strip()
            if text_clean not in seen:
                seen.add(text_clean)
                unique.append(r)
        return unique[:3]

    async def _auto_extract_and_save(self, file_path: str, api_key: str):
        # 1. Read all text from PDF
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
        
        if not full_text.strip():
            return

        # Limit text length to avoid token limits
        full_text = full_text[:8000]

        # 2. Call Gemini
        import google.generativeai as genai
        import json
        genai.configure(api_key=api_key)
        
        prompt = f"""
You are an expert document parser. Analyze the following text extracted from a PDF document (e.g. invoice, ticket, bill, medical report, statement).
Extract all relevant structured information to populate the user's personal memory database:

1. Expenses: any purchases, bills, tickets, payments, or money spent. 
   For each expense, extract:
   - "description": clear description of what was purchased/paid (e.g. "Goa Flight Ticket", "Electricity Bill")
   - "amount": numeric value of the cost (float)
   - "date": date of transaction in YYYY-MM-DD format (if not present, default to today's date)
   - "category": choose from ["Food", "Travel", "Entertainment", "Groceries", "Rent", "Bills", "Shopping", "Other"]

2. Profile Facts: any personal details about the user (e.g., name, college, city, age, job title, pet name, favorite food).
   For each fact, extract:
   - "key": normalized field name (choose from: "name", "college", "city", "birthday", "profession", "favorite food", "pet name")
   - "value": the extracted detail string

3. Notes / Reminders: any important dates, due dates, schedules, plans, policies, or general notes.
   For each note, extract:
   - "content": the text content of the note/reminder (e.g. "EMI due date is 15th of every month")
   - "title": a short descriptive title for the note

Provide the output ONLY as a valid JSON object with the following structure:
{{
  "expenses": [{{"description": "...", "amount": 100.0, "date": "YYYY-MM-DD", "category": "..."}}],
  "profile_facts": [{{"key": "...", "value": "..."}}],
  "notes": [{{"content": "...", "title": "..."}}]
}}

If no data of a category is found, return an empty list for that key.

Extracted PDF text:
---
{full_text}
---
JSON Output:
"""
        loop = asyncio.get_running_loop()
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
        )
        
        text_response = response.text.strip()
        if text_response.startswith("```"):
            lines = text_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text_response = "\n".join(lines).strip()
            
        data = json.loads(text_response)
        
        # 3. Save extracted items
        if self.expense_tool and data.get("expenses"):
            for exp in data["expenses"]:
                await self.expense_tool.add_expense(
                    description=exp.get("description"),
                    amount=float(exp.get("amount", 0)),
                    date_str=exp.get("date"),
                    category=exp.get("category", "General")
                )
        if self.profile_tool and data.get("profile_facts"):
            for fact in data["profile_facts"]:
                self.profile_tool.store_fact(
                    key=fact.get("key"),
                    value=fact.get("value")
                )
        if self.notes_tool and data.get("notes"):
            for note in data["notes"]:
                await self.notes_tool.add_note(
                    content=note.get("content"),
                    title=note.get("title")
                )
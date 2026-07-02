import os
import asyncio
import re
from typing import List, Dict, Any

class PDFTool:
    def __init__(self, db):
        self.db = db

    async def ingest_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF read karo, chunk karo, vectorstore mein daalo."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ingest_pdf_sync, file_path)

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
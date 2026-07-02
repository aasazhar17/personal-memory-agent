import re
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

class AgentPlanner:
    def __init__(self, pdf_tool, expense_tool, notes_tool, calculator_tool, profile_tool, router):
        self.pdf_tool = pdf_tool
        self.expense_tool = expense_tool
        self.notes_tool = notes_tool
        self.calculator_tool = calculator_tool
        self.profile_tool = profile_tool
        self.router = router

    async def execute(self, query: str, api_key: str = None) -> Dict[str, Any]:
        # Parse query: if it has context prefix, extract user query
        user_query = query
        if "\n\nUser Query: " in query:
            parts = query.split("\n\nUser Query: ")
            user_query = parts[1]
        
        # Check for local greeting, chit-chat, identity, or help fallbacks
        chat_fallback = self._handle_local_chat_fallback(user_query)
        if chat_fallback:
            return chat_fallback
        
        steps = []
        tools_used = []
        
        # 1. Notes addition local pattern check
        note_match = re.match(r'^(?:add|save|write)\s+note:\s*(.*)', user_query, re.IGNORECASE)
        if note_match:
            content = note_match.group(1).strip()
            res = await self.notes_tool.add_note(content)
            return {
                "answer": f"Saved note: **{res['note']['title']}**",
                "steps": ["Parsed local add note command"],
                "tools_used": ["notes_tool"]
            }
            
        # 2. Expense addition local pattern check
        exp_match1 = re.match(r'^add\s+expense:\s*(.+?)\s+(\d+(?:\.\d+)?)$', user_query, re.IGNORECASE)
        exp_match2 = re.search(r'(?:spent|paid)\s+(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s+(?:on|for)\s+(.+)', user_query, re.IGNORECASE)
        exp_match3 = re.search(r'(.+?)\s+(?:cost|costs|priced at)\s+(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)$', user_query, re.IGNORECASE)
        
        amount, desc = None, None
        if exp_match1:
            desc = exp_match1.group(1).strip()
            amount = float(exp_match1.group(2))
        elif exp_match2:
            amount = float(exp_match2.group(1))
            desc = exp_match2.group(2).strip()
        elif exp_match3:
            desc = exp_match3.group(1).strip()
            amount = float(exp_match3.group(2))
            
        if amount is not None and desc is not None:
            category = "General"
            for cat in ["food", "travel", "flight", "groceries", "rent", "entertainment", "bills"]:
                if cat in desc.lower() or cat in user_query.lower():
                    category = cat.capitalize()
                    break
            res = await self.expense_tool.add_expense(desc, amount, datetime.now().strftime("%Y-%m-%d"), category)
            return {
                "answer": f"Logged expense: **{desc}** of **₹{amount:.2f}** (Category: {category})",
                "steps": ["Parsed local add expense command"],
                "tools_used": ["expense_tool"]
            }

        # 3. Fact extraction local pattern check
        facts = self._extract_facts(user_query)
        if facts:
            for k, v in facts.items():
                self.profile_tool.store_fact(k, v)
            ans = "Okay, I've saved that in my memory! "
            parts = [f"your {k} is **{v}**" for k, v in facts.items()]
            ans += " and ".join(parts)
            return {
                "answer": ans,
                "steps": [f"Stored facts: {facts}"],
                "tools_used": ["profile_tool"]
            }

        # 4. If we have Gemini api_key, run the LLM agent flow
        if api_key:
            try:
                agent_res = await self._run_llm_agent(user_query, api_key)
                if agent_res:
                    return agent_res
                else:
                    steps.append("LLM agent did not return a valid action structure.")
            except Exception as e:
                import traceback
                print(f"Exception in LLM agent: {str(e)}")
                traceback.print_exc()
                steps.append(f"LLM agent failed with error: {str(e)}")
                
        # 5. Local routing and execution fallback
        tools = await self.router.route(user_query, api_key=None)
        observations = {}
        steps = [f"Routed locally to tools: {tools}"]
        
        for tool_name in tools:
            if tool_name == "profile_tool":
                facts = self.profile_tool.get_all_facts()
                if facts:
                    # Filter facts matching query keywords/synonyms
                    matched_facts = {}
                    synonyms = {
                        "name": ["name"],
                        "college": ["college", "university", "school"],
                        "city": ["city", "live", "location", "hometown"],
                        "favourite food": ["food", "eat", "dish", "favourite", "favorite"],
                        "pet name": ["pet", "dog", "cat", "animal"],
                        "birthday": ["birthday", "born"],
                        "profession": ["profession", "job", "work", "occupation"]
                    }
                    for k, v in facts.items():
                        syms = synonyms.get(k, [k])
                        if any(s in user_query.lower() for s in syms):
                            matched_facts[k] = v
                    
                    if matched_facts:
                        observations["profile"] = matched_facts
                        tools_used.append("profile_tool")
                    else:
                        profile_words = ["profile", "who am i", "about myself", "about me", "know about me", "my facts"]
                        if any(w in user_query.lower() for w in profile_words):
                            observations["profile"] = facts
                            tools_used.append("profile_tool")
            elif tool_name == "pdf_tool":
                results = await self.pdf_tool.search(user_query)
                if results:
                    observations["pdf_tool"] = results
                    tools_used.append("pdf_tool")
            elif tool_name == "expense_tool":
                if "after" in user_query.lower() and "goa" in user_query.lower():
                    note_results = await self.notes_tool.search("Goa trip date")
                    goa_date = None
                    for n in note_results:
                        if "goa" in n.get("content", "").lower():
                            dates = re.findall(r'\d{4}-\d{2}-\d{2}', n.get("content", ""))
                            if dates:
                                goa_date = max(dates)
                                break
                    if goa_date:
                        exp_after = await self.expense_tool.get_expenses_after_date(goa_date)
                        if exp_after:
                            observations["expense_tool"] = exp_after
                            tools_used.append("expense_tool")
                else:
                    results = await self.expense_tool.search(user_query)
                    if results:
                        observations["expense_tool"] = results
                        tools_used.append("expense_tool")
            elif tool_name == "notes_tool":
                results = await self.notes_tool.search(user_query)
                if results:
                    observations["notes_tool"] = results
                    tools_used.append("notes_tool")
            elif tool_name == "calculator_tool":
                expr_match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)', user_query)
                if expr_match:
                    expr = expr_match.group(1).strip()
                    if any(op in expr for op in ["+", "-", "*", "/"]):
                        calc = await self.calculator_tool.calculate(expr)
                        if calc.get("success"):
                            observations["calculator_tool"] = calc
                            tools_used.append("calculator_tool")
                            
        if observations:
            answer = self._synthesize(user_query, observations)
        else:
            answer = (
                "I couldn't find any relevant information in your personal records.\n\n"
                "💡 **Tip**: You can save new information directly in the chat! Try saying:\n"
                "- *\"add note: [note content]\"*\n"
                "- *\"spent 500 on dinner\"*\n"
                "- *\"my college name is [Name]\"*"
            )
            
        return {
            "answer": answer,
            "steps": steps,
            "tools_used": tools_used
        }

    def _handle_local_chat_fallback(self, query: str) -> Dict[str, Any]:
        """Recognize and handle local fallback for greetings, chit-chat, identity, and help."""
        # Normalize
        q = query.lower().strip().strip('?!.')
        words = q.split()
        
        # 1. Greetings (exact match or very short query)
        greetings = {
            "hi", "hello", "hey", "hola", "greetings", "hii", "heyy", "hey there", "hi there", "hello there",
            "good morning", "good afternoon", "good evening", "howdy", "yo"
        }
        if q in greetings:
            return {
                "answer": "Hello! How can I help you today? Ask me about your notes, expenses, documents, or personal profile facts.",
                "steps": ["Parsed local greeting"],
                "tools_used": []
            }
            
        # 2. Chit-chat (exact match or short query containing key phrases)
        chitchat_exact = {
            "how are you", "how's it going", "how are you doing", "what's up", "how have you been", "how is it going",
            "are you there", "are you online"
        }
        chitchat_phrases = ["how are you", "how's it going", "what's up", "how are you doing"]
        if q in chitchat_exact or (len(words) <= 5 and any(phrase in q for phrase in chitchat_phrases)):
            return {
                "answer": "I'm doing well, thank you! I'm here as your Personal Memory Agent, ready to assist you. How can I help you today?",
                "steps": ["Parsed local chit-chat"],
                "tools_used": []
            }
            
        # 3. Identity (asking who the bot is)
        identity_exact = {
            "who are you", "what is your name", "what's your name", "who am i talking to", "what are you", "who are u",
            "what is your identity"
        }
        identity_phrases = ["who are you", "what is your name", "what's your name", "who are u"]
        if q in identity_exact or (len(words) <= 6 and any(phrase in q for phrase in identity_phrases)):
            return {
                "answer": "I am your Personal Memory Agent, a state-of-the-art assistant designed to capture, synthesize, search, and calculate information from your notes, expenses, PDF documents, and profile facts.",
                "steps": ["Parsed local identity query"],
                "tools_used": []
            }
            
        # 4. Capabilities / Help (must be a pure help query, not search query)
        help_exact = {
            "help", "how do you work", "how can you help", "what are your features",
            "capabilities", "features", "how to use", "commands", "menu", "what can you do"
        }
        help_phrases = ["what can you do", "how do you work", "what are your features", "how to use"]
        # If the user literally just typed "help", or a short help phrase
        if q in help_exact or (len(words) <= 5 and any(phrase in q for phrase in help_phrases)):
            help_text = (
                "I can help you with:\n\n"
                "- 📝 **Notes Manager**: Save notes by saying *\"add note: [content]\"* or searching them.\n"
                "- 💰 **Expense Tracker**: Log expenses (e.g., *\"spent 500 on dinner\"*) and view breakdowns.\n"
                "- 👤 **Profile & Documents**: Store profile facts (e.g., *\"my pet name is Rocky\"*) and search uploaded PDFs.\n"
                "- 🧮 **Calculator**: Perform arithmetic queries (e.g., *\"Calculate 12000 + 4500\"*).\n\n"
                "Ask me questions in natural language and I will synthesize the answers!"
            )
            return {
                "answer": help_text,
                "steps": ["Parsed local help/capabilities query"],
                "tools_used": []
            }
            
        return None

    async def _run_llm_agent(self, query: str, api_key: str) -> Dict[str, Any]:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        prompt = f"""
You are the brain of a Personal Memory Assistant.
Your task is to analyze the user's input and decide the best action(s) to take.
You can call one of these specific functions if the user wants to add/save information:
1. `add_note(content: str, title: str = None)` - Use when user wants to save a note, reminder, memo, or note down some text.
2. `add_expense(description: str, amount: float, category: str = "General")` - Use when user wants to log money spent, purchases, pricing.
3. `store_fact(key: str, value: str)` - Use when user shares a personal fact (e.g. my name is Azhar, my dog's name is Rocky).

If the user wants to search, retrieve, or ask a question about their information, you must call:
4. `query_records(search_queries: list)` - Return a list of specific search terms to query the database. For example: ["Goa trip date", "flight cost"]. You should target matching tools.

If the user is just saying hello, chatting, or has a query that doesn't fit the above, call:
5. `chat_response(reply: str)` - Provide a direct friendly response.

Output ONLY a JSON object representing the action and its arguments.
Example 1: {{"action": "add_note", "args": {{"content": "Call mom at 5pm", "title": "Call Mom"}}}}
Example 2: {{"action": "add_expense", "args": {{"description": "Dinner at Pizza Hut", "amount": 450.0, "category": "Food"}}}}
Example 3: {{"action": "store_fact", "args": {{"key": "pet name", "value": "Rocky"}}}}
Example 4: {{"action": "query_records", "args": {{"search_queries": ["Goa trip", "expenses after Goa"]}}}}
Example 5: {{"action": "chat_response", "args": {{"reply": "Hello! How can I help you today?"}}}}

User Input: "{query}"
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
        # Robust JSON extraction
        json_match = re.search(r'(\{.*\})', text_response, re.DOTALL)
        if json_match:
            text_response = json_match.group(1).strip()
        else:
            if text_response.startswith("```"):
                lines = text_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text_response = "\n".join(lines).strip()
            
        data = json.loads(text_response)
        action = data.get("action")
        args = data.get("args", {})
        
        if action == "add_note":
            res = await self.notes_tool.add_note(args.get("content"), args.get("title"))
            return {
                "answer": f"Saved note: **{res['note']['title']}**",
                "steps": [f"LLM agent triggered add_note"],
                "tools_used": ["notes_tool"]
            }
        elif action == "add_expense":
            desc = args.get("description")
            amt = float(args.get("amount", 0))
            cat = args.get("category", "General")
            from datetime import datetime
            res = await self.expense_tool.add_expense(desc, amt, datetime.now().strftime("%Y-%m-%d"), cat)
            return {
                "answer": f"Logged expense: **{desc}** of **₹{amt:.2f}** (Category: {cat})",
                "steps": [f"LLM agent triggered add_expense"],
                "tools_used": ["expense_tool"]
            }
        elif action == "store_fact":
            k = args.get("key")
            v = args.get("value")
            self.profile_tool.store_fact(k, v)
            return {
                "answer": f"Saved fact to profile: **{k}** = **{v}**",
                "steps": [f"LLM agent triggered store_fact"],
                "tools_used": ["profile_tool"]
            }
        elif action == "chat_response":
            return {
                "answer": args.get("reply", "Hello!"),
                "steps": ["LLM agent chat response"],
                "tools_used": []
            }
        elif action == "query_records":
            search_queries = args.get("search_queries", [query])
            observations = {}
            tools_used = []
            steps = [f"LLM generated search queries: {search_queries}"]
            
            for sq in search_queries:
                tools = await self.router.route(sq, api_key)
                for t in tools:
                    if t == "profile_tool":
                        facts = self.profile_tool.get_all_facts()
                        if facts:
                            observations["profile"] = facts
                            if "profile_tool" not in tools_used:
                                tools_used.append("profile_tool")
                    elif t == "pdf_tool":
                        res = await self.pdf_tool.search(sq)
                        if res:
                            if "pdf_tool" not in observations:
                                observations["pdf_tool"] = []
                            for r in res:
                                if r not in observations["pdf_tool"]:
                                    observations["pdf_tool"].append(r)
                            if "pdf_tool" not in tools_used:
                                tools_used.append("pdf_tool")
                    elif t == "expense_tool":
                        if "after" in sq.lower() and "goa" in sq.lower():
                            note_results = await self.notes_tool.search("Goa trip date")
                            goa_date = None
                            for n in note_results:
                                if "goa" in n.get("content", "").lower():
                                    dates = re.findall(r'\d{4}-\d{2}-\d{2}', n.get("content", ""))
                                    if dates:
                                        goa_date = max(dates)
                                        break
                            if goa_date:
                                exp_after = await self.expense_tool.get_expenses_after_date(goa_date)
                                if exp_after:
                                    observations["expense_tool"] = exp_after
                                    if "expense_tool" not in tools_used:
                                        tools_used.append("expense_tool")
                        else:
                            res = await self.expense_tool.search(sq)
                            if res:
                                if "expense_tool" not in observations:
                                    observations["expense_tool"] = []
                                for r in res:
                                    if r not in observations["expense_tool"]:
                                        observations["expense_tool"].append(r)
                                if "expense_tool" not in tools_used:
                                    tools_used.append("expense_tool")
                    elif t == "notes_tool":
                        res = await self.notes_tool.search(sq)
                        if res:
                            if "notes_tool" not in observations:
                                observations["notes_tool"] = []
                            for r in res:
                                if r not in observations["notes_tool"]:
                                    observations["notes_tool"].append(r)
                            if "notes_tool" not in tools_used:
                                tools_used.append("notes_tool")
                    elif t == "calculator_tool":
                        expr_match = re.search(r'([\d\s\+\-\*\/\(\)\.]+)', sq)
                        if expr_match:
                            expr = expr_match.group(1).strip()
                            if any(op in expr for op in ["+", "-", "*", "/"]):
                                calc = await self.calculator_tool.calculate(expr)
                                if calc.get("success"):
                                    observations["calculator_tool"] = calc
                                    if "calculator_tool" not in tools_used:
                                        tools_used.append("calculator_tool")
                                        
            if observations:
                ans = await self._synthesize_with_llm(query, observations, api_key)
            else:
                ans = await self._synthesize_with_llm(query, {}, api_key)
                
            return {
                "answer": ans,
                "steps": steps,
                "tools_used": tools_used
            }
        return None

    async def _synthesize_with_llm(self, query: str, observations: dict, api_key: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        prompt = f"""
You are a helpful Personal Memory Assistant. Your task is to construct a clean, friendly, and natural language response to the user's query using the provided context observations.

User Query: {query}

Observations:
{json.dumps(observations, indent=2)}

Guidelines:
1. Synthesize the observations to answer the query directly and naturally.
2. If the user asks for a total, summary, or calculation, ensure it is accurate based on the observations.
3. Keep the tone friendly, helpful, and conversational.
4. If some context is irrelevant, ignore it.
5. Use markdown formatting to make the answer highly readable (e.g. bold text, bullet points).
6. CRITICAL: If the observations are empty or do not contain the answer, and the user is asking a general knowledge question (e.g., "who is Einstein", "tell me a story", "what is 5+5", "who made you"), answer it using your general knowledge directly.
7. CRITICAL: If the user is asking about their personal data (e.g., "where is my college", "what is my salary", "what is my pet name") and it is not found in the observations, politely inform the user that you couldn't find this information in their saved notes, expenses, or profile facts, and offer to remember it if they share the details.

Response:
"""
        loop = asyncio.get_running_loop()
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )
        return response.text.strip()

    def _extract_facts(self, text: str) -> Dict[str, str]:
        """Extract key-value facts using regex patterns."""
        patterns = {
            "name": r"(?:my name|i am|i'm)\s+(?:is\s+)?([A-Za-z\s]+?)(?:\.|$| and )",
            "college": r"(?:my college|university|school)\s+(?:name\s+)?is\s+([A-Za-z\s]+?)(?:\.|$| and )",
            "city": r"(?:i live in|my city is|i am from)\s+([A-Za-z\s]+?)(?:\.|$| and )",
            "favourite food": r"(?:my favourite|favorite)\s+food\s+is\s+([A-Za-z\s]+?)(?:\.|$| and )",
            "pet name": r"(?:my pet|dog|cat)\s+name\s+is\s+([A-Za-z\s]+?)(?:\.|$| and )",
            "birthday": r"(?:my birthday|born on)\s+is\s+([A-Za-z0-9\s,]+?)(?:\.|$| and )",
            "profession": r"(?:my job|profession|occupation)\s+is\s+([A-Za-z\s]+?)(?:\.|$| and )",
        }
        facts = {}
        parts = re.split(r'\s+and\s+', text)
        for part in parts:
            for key, pat in patterns.items():
                m = re.search(pat, part, re.IGNORECASE)
                if m and key not in facts:
                    facts[key] = m.group(1).strip()
        return facts

    def _synthesize(self, query: str, observations: dict) -> str:
        """Convert observations to readable answer."""
        lines = []
        if "profile" in observations:
            facts = observations["profile"]
            lines.append("📝 **From your profile:**")
            for k, v in facts.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        if "pdf_tool" in observations:
            lines.append("📄 **From uploaded documents:**")
            for r in observations["pdf_tool"]:
                text = r.get("text", "")[:300]
                source = r.get("source", "unknown")
                lines.append(f"- {text} (Source: {source})")
            lines.append("")
        if "expense_tool" in observations:
            lines.append("💰 **From your expenses:**")
            for r in observations["expense_tool"]:
                lines.append(f"- {r.get('description')}: ₹{r.get('amount')} on {r.get('date')}")
            lines.append("")
        if "notes_tool" in observations:
            lines.append("📝 **From your notes:**")
            for r in observations["notes_tool"]:
                lines.append(f"- {r.get('text', '')[:200]}")
            lines.append("")
        if "calculator_tool" in observations:
            calc = observations["calculator_tool"]
            lines.append(f"🧮 **Calculation:** {calc['expression']} = {calc['result']}")
        return "\n".join(lines) if lines else "No relevant information found."
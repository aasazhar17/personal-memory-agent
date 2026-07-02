import asyncio
import os
import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectorstore.faiss_db import FAISSDatabase
from tools.profile_tool import ProfileTool
from tools.notes_tool import NotesTool
from tools.expense_tool import ExpenseTool
from tools.calculator_tool import CalculatorTool
from agent.router import AgentRouter
from agent.planner import AgentPlanner

async def main():
    print("Initializing components...")
    db = FAISSDatabase()
    profile_tool = ProfileTool(db)
    notes_tool = NotesTool(db)
    expense_tool = ExpenseTool(db)
    calculator_tool = CalculatorTool()
    router = AgentRouter()
    
    planner = AgentPlanner(
        pdf_tool=None,
        expense_tool=expense_tool,
        notes_tool=notes_tool,
        calculator_tool=calculator_tool,
        profile_tool=profile_tool,
        router=router
    )
    
    print("\n--- Test 1: Fetching profile info (Local Routing) ---")
    res1 = await planner.execute("What is my name and where do I live?")
    print("Answer:")
    print(res1["answer"])
    print("Steps:", res1["steps"])
    
    print("\n--- Test 2: Multi-hop expense query (Local Routing) ---")
    res2 = await planner.execute("What did I spend after my Goa trip?")
    print("Answer:")
    print(res2["answer"])
    print("Steps:", res2["steps"])
    
    print("\n--- Test 3: Local calculator execution ---")
    res3 = await planner.execute("Calculate 12000 + 4500 + 1800")
    print("Answer:")
    print(res3["answer"])
    print("Steps:", res3["steps"])

    print("\n--- Test 4: Local greeting and chit-chat fallback ---")
    queries = ["how are you", "who are you", "what can you do", "hi there"]
    for q in queries:
        print(f"\nQuery: '{q}'")
        res = await planner.execute(q)
        print("Answer:")
        print(res["answer"])
        print("Steps:", res["steps"])

if __name__ == "__main__":
    asyncio.run(main())

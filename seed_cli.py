import os
import asyncio
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectorstore.faiss_db import FAISSDatabase
from tools.profile_tool import ProfileTool
from tools.notes_tool import NotesTool
from tools.expense_tool import ExpenseTool

async def seed():
    print("Initializing FAISS Database...")
    db = FAISSDatabase()
    db.clear()
    
    # 1. Profile Tool facts
    print("Seeding profile facts...")
    profile = ProfileTool(db)
    profile.store_fact("name", "Azhar")
    profile.store_fact("college", "IIT Bombay")
    profile.store_fact("city", "Mumbai")
    profile.store_fact("pet name", "Rocky")
    profile.store_fact("favorite food", "Biryani")
    profile.store_fact("birthday", "1999-10-15")
    profile.store_fact("profession", "Software Engineer")
    
    # 2. Notes Tool
    print("Seeding notes...")
    notes = NotesTool(db)
    await notes.add_note(
        "Goa trip with friends was from 2026-06-10 to 2026-06-15. We stayed at Candolim beach. Renting a scooty was ₹400/day. Best vacation ever!",
        "Goa Trip 2026"
    )
    await notes.add_note(
        "Home Loan EMI of ₹25000 is due on the 5th of every month. Account: SBI. Autopay is active.",
        "Home Loan EMI Schedule"
    )
    await notes.add_note(
        "Doctor prescription: Take Vitamin D3 weekly on Sunday mornings. Eye drops twice daily.",
        "Medical Prescriptions"
    )
    
    # 3. Expense Tool
    print("Seeding expenses...")
    expenses = ExpenseTool(db)
    # Goa trip expenses
    await expenses.add_expense("Goa Flight Ticket Roundtrip", 8500.0, "2026-06-10", "Travel")
    await expenses.add_expense("Candolim Resort Booking", 12000.0, "2026-06-11", "Travel")
    await expenses.add_expense("Dinner at Thalassa Goa", 3200.0, "2026-06-12", "Food")
    await expenses.add_expense("Scooty Rental and Petrol Goa", 1800.0, "2026-06-14", "Travel")
    # Expenses after Goa
    await expenses.add_expense("Spotify Monthly Subscription", 179.0, "2026-06-20", "Entertainment")
    await expenses.add_expense("Grocery at DMart", 2450.0, "2026-06-22", "Groceries")
    await expenses.add_expense("Internet bill", 999.0, "2026-06-25", "Bills")
    await expenses.add_expense("Dinner with team at Mumbai", 4500.0, "2026-06-28", "Food")
    
    print("\nDatabase seeded successfully!")
    print(f"Total profile facts: {len(profile.get_all_facts())}")
    print(f"Total notes: {len(notes.notes)}")
    print(f"Total expenses: {len(expenses.expenses)}")
    print(f"Total vector store entries: {db.index.ntotal}")

if __name__ == "__main__":
    asyncio.run(seed())

import streamlit as st
import asyncio
import os
import sys
import json
import re
from datetime import datetime
import pandas as pd

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.faiss_db import FAISSDatabase
from tools.profile_tool import ProfileTool
from tools.notes_tool import NotesTool
from tools.pdf_tool import PDFTool
from tools.expense_tool import ExpenseTool
from tools.calculator_tool import CalculatorTool
from agent.router import AgentRouter
from agent.planner import AgentPlanner
from memory.hybrid_memory import HybridMemory

def render_ui():
    st.set_page_config(
        page_title="Personal Memory Agent", 
        page_icon="🧠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom Premium CSS with Google Fonts, glassmorphism, gradients, and micro-interactions
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek gradient background for headers */
    .header-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(77, 150, 255, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
    }
    
    .gradient-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4D96FF 0%, #FF6B6B 50%, #FFD93D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-top: 0.5rem;
    }
    
    /* Glassmorphism card layouts */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(77, 150, 255, 0.3);
        box-shadow: 0 8px 30px rgba(77, 150, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4D96FF;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Interactive custom notes card */
    .note-card {
        border-left: 4px solid #FF6B6B;
        background: rgba(255, 107, 107, 0.03);
        padding: 1rem;
        border-radius: 4px 12px 12px 4px;
        margin-bottom: 0.8rem;
    }
    
    .note-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F7FAFC;
    }
    
    .note-meta {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.2rem;
    }
    
    /* Smooth transition for Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #A0AEC0;
        font-size: 1rem;
        font-weight: 500;
        padding: 10px 16px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #F7FAFC;
        background-color: rgba(77, 150, 255, 0.08);
        border-color: rgba(77, 150, 255, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(77, 150, 255, 0.15) !important;
        border-color: rgba(77, 150, 255, 0.4) !important;
        color: #4D96FF !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

    # Initialize all resources in session state
    if "db" not in st.session_state:
        st.session_state.db = FAISSDatabase()
    if "profile_tool" not in st.session_state:
        st.session_state.profile_tool = ProfileTool(st.session_state.db)
    if "notes_tool" not in st.session_state:
        st.session_state.notes_tool = NotesTool(st.session_state.db)
    if "expense_tool" not in st.session_state:
        st.session_state.expense_tool = ExpenseTool(st.session_state.db)
    if "pdf_tool" not in st.session_state:
        st.session_state.pdf_tool = PDFTool(
            st.session_state.db,
            profile_tool=st.session_state.profile_tool,
            notes_tool=st.session_state.notes_tool,
            expense_tool=st.session_state.expense_tool
        )
    if "calculator_tool" not in st.session_state:
        st.session_state.calculator_tool = CalculatorTool()
    if "router" not in st.session_state:
        st.session_state.router = AgentRouter()
    if "planner" not in st.session_state:
        st.session_state.planner = AgentPlanner(
            pdf_tool=st.session_state.pdf_tool,
            expense_tool=st.session_state.expense_tool,
            notes_tool=st.session_state.notes_tool,
            calculator_tool=st.session_state.calculator_tool,
            profile_tool=st.session_state.profile_tool,
            router=st.session_state.router
        )
    if "memory" not in st.session_state:
        st.session_state.memory = HybridMemory(window_size=5)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Premium App Header
    st.markdown("""
    <div class="header-container">
        <h1 class="gradient-title">🧠 Personal Memory Agent</h1>
        <div class="subtitle-text">A state-of-the-art assistant designed to synthesize notes, expenses, facts, and documents seamlessly.</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Control Center")
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key to activate smart routing and natural language response synthesis.")
        if api_key:
            st.success("API Key detected! AI enhancement active.", icon="⚡")
        else:
            st.info("Using local processing. Add API Key for AI synthesis.", icon="ℹ️")
            
        st.divider()
        st.markdown("### 📊 Database Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Facts Saved", len(st.session_state.profile_tool.get_all_facts()))
            st.metric("Expenses Logged", len(st.session_state.expense_tool.expenses))
        with col2:
            st.metric("Notes Added", len(st.session_state.notes_tool.notes))
            st.metric("Vector Chunks", st.session_state.db.index.ntotal)
            
        st.divider()
        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            st.session_state.db.clear()
            st.session_state.profile_tool.clear()
            st.session_state.notes_tool.notes = []
            st.session_state.notes_tool._save_notes()
            st.session_state.expense_tool.expenses = []
            st.session_state.expense_tool._save_expenses()
            st.session_state.memory.clear()
            st.session_state.chat_history = []
            st.success("All local memory deleted successfully!")
            st.rerun()

    # Premium Multi-Tab Interface
    tab_chat, tab_notes, tab_expenses, tab_profile = st.tabs([
        "💬 Chat Assistant", 
        "📝 Notes Manager", 
        "💰 Expense Tracker", 
        "👤 Profile & Documents"
    ])

    # TAB 1: Chat Assistant
    with tab_chat:
        st.subheader("Conversation Hub")
        
        # Chat log list
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "steps" in msg and msg["steps"]:
                    with st.expander("🔍 Show reasoning path"):
                        for step in msg["steps"]:
                            st.write(step)
                            
        # User input area
        user_query = st.chat_input("Ask about your notes, total cost of travel, details in receipts, or save facts...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Add context from memory window
                    memory_context = st.session_state.memory.get_formatted_context()
                    full_query = f"{memory_context}\nUser Query: {user_query}" if memory_context else user_query
                    
                    try:
                        clean_key = api_key.strip() if api_key else None
                        result = asyncio.run(
                            st.session_state.planner.execute(full_query, api_key=clean_key)
                        )
                        answer = result["answer"]
                        steps = result["steps"]
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        answer = f"⚠️ **Error during execution:** {str(e)}\n\n*Please verify if your Gemini API key is valid and you have a stable network connection. If using an API key, try clearing it from the sidebar to fallback to local offline mode.*"
                        steps = [f"Exception occurred: {str(e)}"]
                    
                    st.write(answer)
                    if steps:
                        with st.expander("🔍 Show reasoning path"):
                            for step in steps:
                                st.write(step)
                                
                    # Update memory logs
                    st.session_state.memory.add_message("user", user_query)
                    st.session_state.memory.add_message("assistant", answer)
                    
                    # Store history for session
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "steps": steps})
                    st.rerun()

    # TAB 2: Notes Manager
    with tab_notes:
        st.subheader("Manage Notes")
        col_add, col_view = st.columns([1, 2])
        
        with col_add:
            st.markdown("#### ➕ Create New Note")
            with st.form("add_note_form", clear_on_submit=True):
                note_title = st.text_input("Title (Optional)")
                note_content = st.text_area("Content")
                submit_note = st.form_submit_button("Save Note", use_container_width=True)
                
                if submit_note and note_content:
                    res = asyncio.run(st.session_state.notes_tool.add_note(note_content, note_title))
                    if res.get("success"):
                        st.success("Note saved successfully!")
                        st.rerun()
                        
        with col_view:
            st.markdown("#### 🔍 Saved Notes")
            search_q = st.text_input("Search notes...", key="note_search_input")
            
            notes_list = st.session_state.notes_tool.notes
            if search_q:
                # Local match filter
                filtered = [n for n in notes_list if search_q.lower() in n["content"].lower() or search_q.lower() in n.get("title", "").lower()]
            else:
                filtered = notes_list
                
            if not filtered:
                st.info("No notes found matching query.")
            else:
                for note in reversed(filtered):
                    st.markdown(f"""
                    <div class="note-card">
                        <div class="note-title">{note.get('title') or 'Untitled Note'}</div>
                        <div class="note-meta">Created on: {note.get('created_at')} | ID: #{note.get('id')}</div>
                        <p style="margin-top: 0.5rem; color: #E2E8F0;">{note.get('content')}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # TAB 3: Expense Tracker
    with tab_expenses:
        st.subheader("Track Expenses")
        
        # Summary KPI metrics
        expenses = st.session_state.expense_tool.expenses
        total_spent = sum(e["amount"] for e in expenses)
        avg_spent = total_spent / len(expenses) if expenses else 0.0
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Total Spend</div>
                <div class="metric-value">₹{total_spent:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Average Transaction</div>
                <div class="metric-value">₹{avg_spent:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Transactions Count</div>
                <div class="metric-value">{len(expenses)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        col_form, col_chart = st.columns([1, 2])
        
        with col_form:
            st.markdown("#### ➕ Log Expense")
            with st.form("add_expense_form", clear_on_submit=True):
                exp_desc = st.text_input("Description")
                exp_amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")
                exp_date = st.date_input("Date", value=datetime.now())
                exp_cat = st.selectbox("Category", ["Food", "Travel", "Entertainment", "Groceries", "Rent", "Bills", "Shopping", "Other"])
                submit_exp = st.form_submit_button("Save Expense", use_container_width=True)
                
                if submit_exp and exp_desc and exp_amount > 0:
                    date_str = exp_date.strftime("%Y-%m-%d")
                    res = asyncio.run(st.session_state.expense_tool.add_expense(exp_desc, exp_amount, date_str, exp_cat))
                    if res.get("success"):
                        st.success("Expense logged!")
                        st.rerun()
                        
        with col_chart:
            st.markdown("#### 📊 Spend Breakdown by Category")
            if expenses:
                df = pd.DataFrame(expenses)
                cat_summary = df.groupby("category")["amount"].sum().reset_index()
                st.bar_chart(cat_summary.set_index("category"))
            else:
                st.info("Log some expenses to view charts.")
                
        st.divider()
        st.markdown("#### 📋 Transaction History")
        if expenses:
            df = pd.DataFrame(expenses)
            # Reorder columns for presentation
            df = df[["id", "date", "description", "category", "amount"]]
            df = df.rename(columns={"id": "ID", "date": "Date", "description": "Description", "category": "Category", "amount": "Amount (₹)"})
            st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No transaction history available.")

    # TAB 4: Profile & Documents
    with tab_profile:
        st.subheader("Profile & Ingested Files")
        
        col_prof, col_files = st.columns(2)
        
        with col_prof:
            st.markdown("#### 👤 My Profile Facts")
            facts = st.session_state.profile_tool.get_all_facts()
            
            with st.form("add_fact_form", clear_on_submit=True):
                fact_key = st.selectbox("Fact Field", ["Name", "College", "City", "Birthday", "Profession", "Favorite Food", "Pet Name"])
                fact_val = st.text_input("Value")
                submit_fact = st.form_submit_button("Save Fact", use_container_width=True)
                
                if submit_fact and fact_val:
                    st.session_state.profile_tool.store_fact(fact_key, fact_val)
                    st.success("Profile fact updated!")
                    st.rerun()
                    
            if facts:
                fact_data = [{"Fact Field": k.title(), "Saved Value": v} for k, v in facts.items()]
                st.table(pd.DataFrame(fact_data))
            else:
                st.info("No profile facts stored yet.")
                
        with col_files:
            st.markdown("#### 📁 PDF Ingestion Hub")
            uploaded_file = st.file_uploader("Index a new PDF document into local memory", type=["pdf"])
            if uploaded_file is not None:
                docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
                os.makedirs(docs_dir, exist_ok=True)
                temp_path = os.path.join(docs_dir, uploaded_file.name)
                
                # Check if it was already uploaded in this session run to prevent re-runs
                if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"Reading and indexing {uploaded_file.name}..."):
                        clean_key = api_key.strip() if api_key else None
                        res = asyncio.run(st.session_state.pdf_tool.ingest_pdf(temp_path, api_key=clean_key))
                        if res.get("success"):
                            st.success(res.get("message"))
                            st.session_state.last_uploaded_name = uploaded_file.name
                        else:
                            st.error(res.get("message"))
                            
            # List ingested documents
            st.markdown("#### 📄 Ingested Document List")
            vector_items = st.session_state.db.metadatas
            unique_pdfs = set()
            for meta in vector_items:
                if meta.get("type") == "pdf" and meta.get("source"):
                    unique_pdfs.add(meta.get("source"))
                    
            if unique_pdfs:
                for pdf in unique_pdfs:
                    st.markdown(f"- 📄 **{pdf}**")
            else:
                st.info("No documents indexed in database yet.")

if __name__ == "__main__":
    render_ui()
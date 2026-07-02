import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.streamlit_ui import render_ui

if __name__ == "__main__":
    render_ui()

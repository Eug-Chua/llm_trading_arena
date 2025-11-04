"""DOGE Analysis Page"""
import streamlit as st
from pathlib import Path
import sys
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.generic_coin_page import render_coin_page

# Load DOGE logo for page icon
logo_path = project_root / "frontend" / "images" / "doge.png"
page_icon = Image.open(logo_path) if logo_path.exists() else "🐕"

st.set_page_config(page_title="DOGE Analysis", page_icon=page_icon, layout="wide")

render_coin_page("DOGE")

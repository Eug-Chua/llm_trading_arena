"""BTC Analysis Page"""
import streamlit as st
from pathlib import Path
import sys
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.generic_coin_page import render_coin_page

# Load BTC logo for page icon
logo_path = project_root / "frontend" / "images" / "coin_logos" / "btc.png"
page_icon = Image.open(logo_path) if logo_path.exists() else "₿"

st.set_page_config(page_title="BTC Analysis", page_icon=page_icon, layout="wide")

render_coin_page("BTC")

"""SOL Analysis Page"""
import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.generic_coin_page import render_coin_page

st.set_page_config(page_title="SOL Analysis", layout="wide")

render_coin_page("SOL")

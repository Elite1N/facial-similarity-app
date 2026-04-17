# app/main.py (Example for frontend)
import streamlit as st
from src.interface import get_match

uploaded_file = st.file_uploader("Upload a selfie")
if uploaded_file:
    # Save (in mem) temp file and match
    result = get_match(uploaded_file)
    st.write(f"You look like: {result['name']}")
    st.image(result['image_path'])
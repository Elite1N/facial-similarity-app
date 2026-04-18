# Starter code (???)
# app/main.py (Example for frontend)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.interface import get_match

uploaded_file = st.file_uploader("Upload a selfie")
if uploaded_file:
    # Save (in mem) temp file and match
    result = get_match(uploaded_file)
    print(result)
    st.write(f"You look like: {result['Name']}")
    st.image(result['Image Path'])
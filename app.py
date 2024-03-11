import streamlit as st
import google.generativeai  as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key="GOOGLE_API_KEY")





# streamlit application
st.set_page_config("RAG system")
st.markdown("<h2 style='text-align: center; color: Black;'>RAG-based Question Answering System</h2>", unsafe_allow_html=True)
# get_translations.py
import json
import streamlit as st

@st.cache_data
def cargar_idioma(idioma="es"):
    try:
        with open(f"translations/{idioma}.json", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        with open("translations/es.json", encoding='utf-8') as f:
            return json.load(f)

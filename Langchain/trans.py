# streamlit_ollama_chatbot_jsonlog.py
# -*- coding: utf-8 -*-

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Seiteneinstellungen ---
st.set_page_config(page_title="Chatbot mit JSON-Verlauf", page_icon="🧾", layout="centered")

st.title("🧾 Zweisprachiger Chatbot mit JSON-Verlauf")

st.markdown(
    """
    Sprich mit dem Bot auf **Englisch** oder **Französisch** – er antwortet **immer auf Deutsch**  
    und speichert jedes Gespräch mit **Datum & Uhrzeit** in einer JSON-Datei.
    """
)

# --- Speicherpfad ---
LOG_FILE = Path("chat_history.json")

def append_to_json(role: str, content: str):
    """Schreibt eine Chatnachricht in eine JSON-Datei."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "role": role, "content": content}

    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- Sprache auswählen ---
language = st.selectbox("Welche Sprache möchtest du verwenden?", ("Englisch", "Französisch"))

if language == "Englisch":
    system_prompt = (
        "You are a bilingual assistant. The user will speak English. "
        "Understand their message, remember context, and respond in fluent, natural German. "
        "Do not repeat the English text, only give the German translation and answer."
    )
else:
    system_prompt = (
        "You are a bilingual assistant. The user will speak French. "
        "Understand their message, remember context, and respond in fluent, natural German. "
        "Do not repeat the French text, only give the German translation and answer."
    )

# --- Modell + Chain ---
model = ChatOllama(model="llama3.1")
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{text}")
])
chain = prompt | model | parser

# --- Memory-Setup ---
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="text",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "default_session"}}

# --- Verlauf anzeigen ---
history = get_session_history("default_session")
for msg in history.messages:
    if msg.type == "human":
        st.markdown(f"🧑‍💻 **Du:** {msg.content}")
    elif msg.type == "ai":
        st.markdown(
            f"<div style='background-color:#e6f7ff; padding:10px; border-radius:8px;'>🤖 <b>Bot:</b> {msg.content}</div>",
            unsafe_allow_html=True,
        )

# --- Eingabe ---
user_input = st.chat_input("Frag mich etwas auf Englisch oder Französisch...")

# --- Verarbeitung ---
if user_input:
    append_to_json("user", user_input)
    with st.spinner("Denke nach..."):
        response = chat_with_memory.invoke({"text": user_input}, config=config)
    append_to_json("assistant", response)
    st.rerun()
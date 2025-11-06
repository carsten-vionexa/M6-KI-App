# app.py
import streamlit as st
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from rag.retriever import get_relevant_docs
from rag.qa_chain import answer_question
from rag.chat_chain import create_chat_chain

# --------------------------------------------------------------------
# Basis-Pfade
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

# --------------------------------------------------------------------
# Streamlit-Setup
st.set_page_config(page_title="Hawkins Consulting RAG", page_icon="📚")
st.title("📚 Hawkins Consulting – Seminar-RAG")

# --------------------------------------------------------------------
# Session-State initialisieren
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = create_chat_chain()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------
# Eingabe
query = st.text_input("Frage oder Folgefrage eingeben:")

# --------------------------------------------------------------------
# Anfrage-Button
if st.button("Antwort anzeigen") and query:
    with st.spinner("🔍 Suche nach relevanten Inhalten ..."):
        # 1. Embeddings + Vektordatenbank laden
        embedder = OllamaEmbeddings(model="mxbai-embed-large")
        db = Chroma(persist_directory=str(CHROMA_PATH),
                    embedding_function=embedder)

        # 2. Relevante Dokumente suchen
        docs = get_relevant_docs(db, query, k=6)

        # 3. Verlauf als Text zusammenstellen
        history_text = "\n".join(
            [f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history]
        )

        # 4. Antwort erzeugen
        chain = st.session_state.chat_chain
        answer = chain(f"{query}\n\nKontext:\n{''.join(d.page_content for d in docs)}",
                       history_text)

        # 5. Verlauf speichern
        st.session_state.chat_history.append((query, answer))

        # 6. Anzeige
        st.subheader("🧠 Antwort")
        st.write(answer)

        st.markdown("---")
        st.subheader("💬 Gesprächsverlauf")
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**{i}.** 🗣️ *{q}*")
            st.markdown(f" → {a}")

        st.markdown("---")
        st.subheader("📄 Quellen")
        for i, d in enumerate(docs, 1):
            src = Path(d.metadata.get("source", "Unbekannt")).name
            page = d.metadata.get("page", "–")
            st.markdown(f"**{i}.** {src}, Seite {page}")

# --------------------------------------------------------------------
# Reset-Button
if st.button("🧹 Neues Gespräch"):
    st.session_state.chat_history = []
    st.rerun()
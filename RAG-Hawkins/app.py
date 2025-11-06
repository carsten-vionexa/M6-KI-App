# app.py
import streamlit as st
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
from langchain_community.vectorstores import Chroma

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
        db = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embedder
        )

        # 2. Relevante Dokumente suchen
        docs = get_relevant_docs(db, query, k=6)

        # 3. Verlauf als Text zusammenstellen
        history_text = "\n".join(
            [f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history]
        )

       # 4. Antwort erzeugen mit Kontext + Memory
        chain = st.session_state.chat_chain
        context_text = "\n".join([d.page_content for d in docs])

        # Verlauf vorher berechnen
        history_formatted = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history])

        conversation_input = f"""
        Bisheriges Gespräch:
        {history_formatted}

        Relevanter Kontext aus den Dokumenten:
        {context_text}

        Neue Frage:
        {query}
        """

        response = chain.invoke(
            {"input": conversation_input},
            config={"configurable": {"session_id": "default"}}
        )

        # 5. Antworttext extrahieren
        answer_text = response.content if hasattr(response, "content") else str(response)

        # 6. Verlauf speichern
        st.session_state.chat_history.append((query, answer_text))

        # 7. Anzeige
        st.subheader("🧠 Antwort")
        st.markdown(answer_text)

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
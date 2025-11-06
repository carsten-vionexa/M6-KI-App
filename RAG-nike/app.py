# app.py
import streamlit as st
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from rag.retriever import get_relevant_docs
from rag.chat_chain import create_chat_chain

# --------------------------------------------------------------------
# Basis-Pfade
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

# --------------------------------------------------------------------
# Streamlit-Setup
st.set_page_config(page_title="Nike 10-K Report – RAG Explorer", page_icon="👟")
st.title("👟 Nike 10-K Report – RAG Explorer")

# --------------------------------------------------------------------
# Session-State initialisieren
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = create_chat_chain()
if "session_id" not in st.session_state:
    # einfache, stabile Session-ID pro Browser-Session
    st.session_state.session_id = "ui-session-1"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------
# Eingabe
query = st.text_input("Frage oder Folgefrage eingeben:")

# --------------------------------------------------------------------
# Anfrage-Button
if st.button("Antwort anzeigen") and query:
    with st.spinner("🔍 Suche nach relevanten Inhalten ..."):
        # 1️⃣ Embeddings + Vektordatenbank laden (muss mit init_db.py gebaut sein)
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedder)

        # 2️⃣ Relevante Dokumente suchen
        docs = get_relevant_docs(db, query, k=6)

        # 3️⃣ Kontext zusammenstellen (inkl. knapper Quellmarker)
        ctx_chunks = []
        for d in docs:
            src = Path(d.metadata.get("source", "n/a")).name
            page = d.metadata.get("page", "–")
            section = d.metadata.get("section", "")
            header = f"[{src} • S.{page}{' • ' + section if section else ''}]"
            ctx_chunks.append(f"{header}\n{d.page_content}")
        context = "\n\n---\n\n".join(ctx_chunks)

        # 4️⃣ Antwort generieren (mit Memory)
        chain = st.session_state.chat_chain
        result = chain.invoke(
            {"question": query, "context": context},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )

        # Modellantwort als Text isolieren
        answer_text = getattr(result, "content", str(result))

        # 5️⃣ Verlauf speichern
        st.session_state.chat_history.append((query, answer_text))

        # 6️⃣ Anzeige
        st.subheader("🧠 Antwort")
        st.write(answer_text)

        st.markdown("---")
        st.subheader("💬 Gesprächsverlauf")
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**{i}.** 🗣️ *{q}*")
            st.markdown(f" → {a}")

        # 7️⃣ Quellen mit Abschnittsinfo
        st.markdown("---")
        st.subheader("📄 Quellen")
        for i, d in enumerate(docs, 1):
            src = Path(d.metadata.get("source", "Unbekannt")).name
            page = d.metadata.get("page", "–")
            section = d.metadata.get("section", "—")
            st.markdown(f"**{i}. _{section}_ — Seite {page}**  \n({src})")

# --------------------------------------------------------------------
# Reset-Button
if st.button("🧹 Neues Gespräch"):
    st.session_state.chat_history = []
    st.rerun()
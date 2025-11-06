# -*- coding: utf-8 -*-
import streamlit as st
import sqlite3
import pandas as pd
import os
from pathlib import Path
import base64

DB_FILE = "papers.db"

st.set_page_config(page_title="Hugging Face Paper Monitor", layout="wide")
st.title("📚 Hugging Face Paper Monitor")
st.caption("Anzeige aller gespeicherten Paper – sortiert nach Datum (neueste zuerst)")

# ===============================
# 1️⃣ Datenbank laden
# ===============================
if not os.path.exists(DB_FILE):
    st.warning("Keine Datenbank gefunden. Bitte zuerst das Sammel-Script ausführen.")
    st.stop()

conn = sqlite3.connect(DB_FILE)
df = pd.read_sql_query("""
    SELECT id, title, authors, summary, date_processed
    FROM papers
    ORDER BY date_processed DESC
""", conn)
conn.close()

if df.empty:
    st.info("Noch keine Paper in der Datenbank.")
    st.stop()

# ===============================
# 2️⃣ Hilfsfunktionen
# ===============================

def make_pdf_link(paper_id):
    """Erzeugt Download-Link für lokale PDF oder Fallback zu arXiv."""
    pdf_path = Path(f"paper_{paper_id}.pdf")
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64 = base64.b64encode(pdf_data).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{pdf_path.name}">📄 Lokales PDF</a>'
    else:
        return f'<a href="https://arxiv.org/pdf/{paper_id}.pdf" target="_blank">🌐 arXiv PDF</a>'

def make_title_link(row):
    """Verlinkt Titel immer auf die arXiv-Seite."""
    return f'<a href="https://arxiv.org/abs/{row["id"]}" target="_blank">{row["title"]}</a>'

# ===============================
# 3️⃣ Tabelle vorbereiten
# ===============================
df["Titel"] = df.apply(make_title_link, axis=1)
df["PDF"] = df["id"].apply(make_pdf_link)

df_display = df[["date_processed", "Titel", "authors", "PDF"]].rename(columns={
    "date_processed": "Datum",
    "authors": "Autoren"
})

# ===============================
# 4️⃣ Tabelle anzeigen
# ===============================
st.markdown(
    df_display.to_html(escape=False, index=False),
    unsafe_allow_html=True
)

# ===============================
# 5️⃣ Zusammenfassungen (optional)
# ===============================
with st.expander("🧠 Zusammenfassungen anzeigen"):
    for _, row in df.iterrows():
        st.markdown(f"### [{row['title']}](https://arxiv.org/abs/{row['id']})")
        st.markdown(f"**Autoren:** {row['authors']}")
        st.markdown(f"**Datum:** {row['date_processed']}")
        st.write(row["summary"])
        pdf_path = Path(f"paper_{row['id']}.pdf")
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Lokales PDF herunterladen",
                    data=f.read(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )
        else:
            st.markdown(f"[🌐 arXiv PDF öffnen](https://arxiv.org/pdf/{row['id']}.pdf)")
        st.divider()
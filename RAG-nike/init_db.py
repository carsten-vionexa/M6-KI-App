# init_db.py
"""
Baut die Chroma-Datenbank aus der Nike 10-K PDF neu auf.
Löscht bestehende DB, liest die PDF, ergänzt Abschnitts-Metadaten und erzeugt Embeddings.
"""

from pathlib import Path
import shutil

from rag.loader import load_pdfs
from rag.semantics import extract_sections, dedupe_by_title
from rag.embedder import get_embedder
from rag.vectorstore import create_or_load_chroma
from langchain_community.vectorstores import Chroma

# --------------------------------------------------------------------
# Basisverzeichnisse
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"

# --------------------------------------------------------------------
# Alte DB löschen (sauberer Neuaufbau)
if CHROMA_PATH.exists():
    print("🧹 Entferne alte Chroma-Datenbank ...")
    shutil.rmtree(CHROMA_PATH)

# --------------------------------------------------------------------
# PDFs laden (Seiten → Chunks)
print("📁 Lade PDF-Dokumente ...")
docs = load_pdfs(str(PDF_PATH), chunk_size=1800, chunk_overlap=150)
print(f"✅ {len(docs)} Dokument-Chunks geladen.")

# --------------------------------------------------------------------
# Abschnitts-Erkennung (10-K typische Überschriften)
print("🔎 Erkenne Abschnitte ...")
sections = dedupe_by_title(extract_sections(docs))

# Mapping Seite → erster passender Abschnittstitel
page_to_section = {}
for s in sections:
    page_raw = s.get("page")
    try:
        page = int(page_raw)
    except Exception:
        continue
    page_to_section.setdefault(page, s["title"])

def assign_section(p: int) -> str:
    # Nächsten Abschnitt mit Startseite ≤ p finden (falling back)
    candidates = [pg for pg in page_to_section.keys() if pg <= p]
    if not candidates:
        return "Item 1. Business"
    start = max(candidates)
    return page_to_section[start]

# Metadaten ergänzen
for d in docs:
    try:
        p = int(d.metadata.get("page", 0))
    except Exception:
        p = 0
    d.metadata["section"] = assign_section(p)

# --------------------------------------------------------------------
# Embeddings & Chroma
print("🧠 Erstelle Embeddings mit OpenAI ...")
embedder = get_embedder()  # text-embedding-3-large
print(f"🔗 Verwende Embedding-Modell: {embedder.model if hasattr(embedder, 'model') else 'OpenAI'}")

print("💾 Erstelle neue Chroma-Datenbank ...")
_ = create_or_load_chroma(docs, embedder, persist_directory=str(CHROMA_PATH))
print(f"✅ Chroma-DB gespeichert unter {CHROMA_PATH}")
print("🎉 Fertig! Die Vektordatenbank wurde neu erstellt.")

# Kurzer Check
print("\n🔎 Überprüfe gespeicherte Metadaten ...")
db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedder)
sample = db.similarity_search("employees headcount", k=10)
ok = sum(1 for d in sample if "section" in d.metadata)
for i, d in enumerate(sample[:10], 1):
    print(f"{i:02d}. Seite: {d.metadata.get('page','?')}, Abschnitt: {d.metadata.get('section','?')}")
print(f"\n✅ {ok} von {len(sample[:10])} Einträgen enthalten 'section'-Metadaten.")
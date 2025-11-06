# init_db.py
"""
Einmaliger Import der Seminar-PDFs in die Chroma-Datenbank.
Löscht bestehende Datenbank, liest alle PDFs neu ein und erstellt Embeddings.
"""

from pathlib import Path
import shutil
from rag.loader import load_pdfs
from rag.embedder import get_embedder
from rag.vectorstore import create_or_load_chroma

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
# PDFs laden
print("📁 Lade PDF-Dokumente ...")
docs = load_pdfs(str(PDF_PATH))
print(f"✅ {len(docs)} Dokument-Chunks geladen.")

# --------------------------------------------------------------------
# Embeddings erzeugen
print("🧠 Erstelle Embeddings mit Open AI ...")
embedder = get_embedder()  
print(f"🧠 Erstelle Embeddings mit {embedder.model}")

# --------------------------------------------------------------------
# Chroma-DB aufbauen
print("💾 Erstelle neue Chroma-Datenbank ...")
create_or_load_chroma(docs, embedder, persist_directory=str(CHROMA_PATH))

print("🎉 Fertig! Die Vektordatenbank wurde neu erstellt.")
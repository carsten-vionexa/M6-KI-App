# rag/loader.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def _load_pdf_pages(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # eine Document-Instanz pro Seite
    # Quelle sauber setzen
    for p in pages:
        p.metadata.setdefault("source", file_path)
    return pages

def load_pdfs(pdf_dir: str, chunk_size: int = 1800, chunk_overlap: int = 150) -> list[Document]:
    pdf_dir = Path(pdf_dir)
    all_pages: list[Document] = []

    for file in sorted(pdf_dir.glob("*.pdf")):
        print(f"📄 Lade {file.name} ...")
        pages = _load_pdf_pages(str(file))
        print(f"✅ {len(pages)} Seiten geladen aus {file.name}")
        all_pages.extend(pages)

    print(f"✂️  Splitte Dokumente in Chunks (size={chunk_size}, overlap={chunk_overlap}) ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_pages)
    print(f"✅ {len(chunks)} Text-Chunks erstellt.")
    return chunks
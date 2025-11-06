# rag/splitter.py
"""
Dynamische Text-Splitting-Optionen für PDF- und Web-Dokumente.
Unterstützt drei Modi: 'classic', 'semantic', 'hybrid'.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def split_docs(docs: list[Document], mode: str = "classic") -> list[Document]:
    """
    Teilt Dokumente in Chunks auf – je nach gewähltem Modus:
      classic  -> feste Zeichenlängen
      semantic -> thematische Brüche über Embeddings
      hybrid   -> Kombination: erst grob, dann semantisch
    """
    print(f"✂️  Split-Modus: {mode}")

    if mode not in {"classic", "semantic", "hybrid"}:
        raise ValueError("Mode muss 'classic', 'semantic' oder 'hybrid' sein.")

    if mode == "classic":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)

    embedder = OllamaEmbeddings(model="mxbai-embed-large")

    if mode == "semantic":
        sem_splitter = SemanticChunker(
            embeddings=embedder,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.4,
        )
        refined = []
        for d in docs:
            refined.extend(sem_splitter.create_documents([d.page_content]))
        return refined

    if mode == "hybrid":
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=0)
        sem_splitter = SemanticChunker(
            embeddings=embedder,
            breakpoint_threshold_amount=0.4)
        refined = []
        for d in base_splitter.split_documents(docs):
            refined.extend(sem_splitter.create_documents([d.page_content]))
        return refined
# rag/retriever.py
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_relevant_docs(db: Chroma, query: str, k: int = 6) -> list[Document]:
    """
    MMR-Retrieval mit leicht erhöhter Breite, um Diversität zu sichern.
    """
    # fetch_k etwas größer lassen, um MMR wirken zu lassen
    return db.max_marginal_relevance_search(query, k=k, fetch_k=30)
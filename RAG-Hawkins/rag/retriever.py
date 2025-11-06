# rag/retriever.py
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_relevant_docs(db: Chroma, query: str, k: int = 6, bm25=None, k_bm25: int = 8) -> List[Document]:
    vdocs = db.max_marginal_relevance_search(query, k=k, fetch_k=24)
    if bm25 is None:
        return vdocs
    bdocs = bm25.invoke(query)[:k_bm25]
    # Merge simpel per Text-Hash
    seen, merged = set(), []
    for d in vdocs + bdocs:
        h = (d.page_content[:120], d.metadata.get("source",""), d.metadata.get("page",""))
        if h not in seen:
            seen.add(h); merged.append(d)
    return merged[:max(k, k_bm25)]
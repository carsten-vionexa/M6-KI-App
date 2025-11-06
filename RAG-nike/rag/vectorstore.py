# rag/vectorstore.py
from langchain_community.vectorstores import Chroma

def create_or_load_chroma(docs, embedder, persist_directory: str = "chroma_db", batch_size: int = 32):
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    db = None

    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        bm = metas[i:i+batch_size]
        if db is None:
            db = Chroma.from_texts(bt, embedder, metadatas=bm, persist_directory=persist_directory)
        else:
            db.add_texts(bt, metadatas=bm)

    # Chroma >=0.4 persisted automatisch; persist() ist ok, auch wenn deprecated
    try:
        db.persist()
    except Exception:
        pass
    return db
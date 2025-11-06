# rag/vectorstore.py
from langchain_community.vectorstores import Chroma

def create_or_load_chroma(docs, embedder, persist_directory="chroma_db", batch_size=16):
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    db = None

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        if db is None:
            db = Chroma.from_texts(batch_texts, embedder, metadatas=batch_metas, persist_directory=persist_directory)
        else:
            db.add_texts(batch_texts, metadatas=batch_metas)
    #db.persist()
    return db
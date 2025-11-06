# rag/embedder.py
from langchain_openai import OpenAIEmbeddings

def get_embedder(model_name: str = "text-embedding-3-large") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name)
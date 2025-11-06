# rag/embedder.py
from langchain_ollama import OllamaEmbeddings

def get_embedder(model_name="mxbai-embed-large"):
    return OllamaEmbeddings(model=model_name)
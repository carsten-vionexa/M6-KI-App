# rag/embedder.py
from langchain_openai import OpenAIEmbeddings

def get_embedder(model_name="text-embedding-3-large"):
    return OpenAIEmbeddings(model=model_name)
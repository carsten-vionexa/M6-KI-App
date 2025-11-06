# rag/loader.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(pdf_dir: str):
    pdf_dir = Path(pdf_dir)
    docs = []
    for file in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

from langchain_community.document_loaders import WebBaseLoader
from requests.exceptions import RequestException

def load_web(urls: list[str]):
    docs = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Hawkins-RAG/1.0; +https://hawkins-consulting.com)"}

    for url in urls:
        try:
            print(f"🌐 Lade {url} ...")
            loader = WebBaseLoader([url])
            loader.requests_kwargs = {"headers": headers, "timeout": 15}
            _docs = loader.load()
            for d in _docs:
                d.metadata.setdefault("source", url)
                d.metadata.setdefault("page", "web")
            docs.extend(_docs)
            print(f"✅ {len(_docs)} Abschnitte geladen von {url}")
        except RequestException as e:
            print(f"⚠️  Fehler beim Laden {url}: {e}")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)
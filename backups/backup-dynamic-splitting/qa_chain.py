# rag/qa_chain.py

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

SYSTEM_PROMPT = """DDu bist ein sachlicher Assistent, der ausschließlich aus dem gegebenen Kontext antwortet.
Verwende keine allgemeinen Phrasen, keine Höflichkeitsfloskeln und keine Fragen an den Nutzer.
Gib nur die geforderte Information wieder.
Wenn die Antwort nicht eindeutig im Kontext steht, schreibe: "Ich weiß es nicht."
Antworte präzise, kurz und füge Quellen in der Form [Quelle: <Datei>, S.<Seite>] hinzu."""

USER_PROMPT = """{system}

Frage:
{question}

Kontext (mehrere Ausschnitte, jeweils mit Quelle und Seite):
{context}

Antworte auf Deutsch, maximal 3 Sätze mit Quellenangaben.
"""

def _format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "Unbekannt")
        page = d.metadata.get("page", "–")
        snippet = d.page_content.replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600] + " …"
        parts.append(f"[Quelle: {src}, S.{page}] {snippet}")
    return "\n\n".join(parts)

def answer_question(query: str, docs: List[Document], model_name="llama3.1") -> str:
    llm = OllamaLLM(model=model_name, temperature=0, num_ctx=8192)
    prompt = PromptTemplate.from_template(USER_PROMPT)
    context = _format_context(docs)
    full_prompt = prompt.format(system=SYSTEM_PROMPT, question=query, context=context)
    return llm.invoke(full_prompt)
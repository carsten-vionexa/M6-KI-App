# rag/semantics.py
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import Ollama
import re, json

def build_bm25(docs: List[Document]) -> BM25Retriever:
    ret = BM25Retriever.from_documents(docs)
    ret.k = 12
    return ret

TITLE_PATTERNS = [
    r"(?im)^\s*(?:seminar|training|workshop)\s*[:\-–]\s*(.+)$",
    r"(?im)^\s*([A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9 &/\-]{6,})\s*$",  # fette Überschrift
]
META_PAT = r"(?i)(dauer|termine?|zielgruppe|inhalte?|voraussetzungen?)"

def extract_candidates_rule(docs: List[Document]) -> List[Dict]:
    items = []
    for d in docs:
        text = d.page_content
        title = None
        for pat in TITLE_PATTERNS:
            m = re.search(pat, text)
            if m:
                title = m.group(1).strip() if m.lastindex else m.group(0).strip()
                break
        if not title:
            # zweite Chance: erste Zeile bis 90 Zeichen als Titel
            first = text.strip().splitlines()[0][:90]
            if len(first) > 12: title = first
        if title and re.search(META_PAT, text):
            items.append({
                "title": re.sub(r"\s{2,}", " ", title),
                "snippet": " ".join(text.strip().split())[:400],
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page", ""),
            })
    return items

LLM_SCHEMA = """Gib nur JSON-Liste aus:
[{"title": "...","dauer": "...","zielgruppe": "...","kurzbeschreibung":"..."}]
Wenn Angaben fehlen, Felder weglassen. Kein Fließtext neben dem JSON!"""

def extract_candidates_llm(docs: List[Document], model="llama3.1") -> List[Dict]:
    llm = Ollama(model=model, temperature=0)
    merged = "\n\n---\n\n".join(
        f"[{Path(d.metadata.get('source','?')).name} S.{d.metadata.get('page','-')}]\n{d.page_content[:1800]}"
        for d in docs[:8]  # Längenlimit
    )
    prompt = (
        "Extrahiere Seminar-Einträge (Titel, ggf. Dauer, Zielgruppe, Kurzbeschreibung) aus den Auszügen.\n"
        f"{LLM_SCHEMA}\n\nAusschnitte:\n{merged}\n\nJSON:"
    )
    out = llm.invoke(prompt).strip()
    try:
        data = json.loads(out)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def dedupe_by_title(items: List[Dict]) -> List[Dict]:
    seen, res = set(), []
    for it in items:
        t = it.get("title","").strip().lower()
        if t and t not in seen:
            seen.add(t); res.append(it)
    return res
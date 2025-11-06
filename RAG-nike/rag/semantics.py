# rag/semantics.py
"""
Erkennt typische 10-K Abschnittstitel (z. B. 'Item 1. Business') und liefert sie mit Seitenangabe.
"""

import re
from typing import List, Dict
from langchain_core.documents import Document

# Robuster Matcher für 10-K Überschriften
SECTION_RE = re.compile(
    r"(?im)^\s*(ITEM\s+\d{1,2}[A-Z]?(?:\.)?\s*[A-Z0-9 ,&/\-\(\)']{3,120})\s*$"
)

def extract_sections(docs: List[Document]) -> List[Dict]:
    sections: List[Dict] = []
    for d in docs:
        # nur die ersten ~1000 Zeichen einer Seite scannen
        head = d.page_content[:1000]
        for m in SECTION_RE.finditer(head):
            raw = m.group(1)
            cleaned = (
                raw.replace("\n", " ")
                   .replace("  ", " ")
                   .strip(" .,:;–—")
            )
            # Normalisierung: „Item 1. Business …“ → bis zu zwei Wörter nach der Nummer
            # 1) „Item X.“ + erstes Wort
            m2 = re.match(r"(?i)(item\s+\d{1,2}[A-Z]?\.?)\s+([A-Za-z][A-Za-z\-&/]*)", cleaned)
            if m2:
                cleaned = f"{m2.group(1).title()} {m2.group(2).title()}"
            else:
                cleaned = cleaned.title()

            sections.append({
                "title": cleaned,
                "page": d.metadata.get("page", "?"),
                "source": d.metadata.get("source", ""),
            })
    return sections

def dedupe_by_title(items: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for it in items:
        t = it.get("title", "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(it)
    return out
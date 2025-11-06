from pathlib import Path

readme_content = """# 📘 RAG-Hawkins – Seminar-RAG-System

## 🔍 Projektbeschreibung
Dieses Projekt implementiert ein eigenes **RAG-System (Retrieval-Augmented Generation)** für *Hawkins Consulting*.
Ziel: Fragen zu Seminaren und Trainings auf Basis interner PDF-Dokumente (z. B. Prospekte, Themenübersichten) automatisch beantworten.

Das System kombiniert:
- **Chroma Vectorstore** für Dokument-Suche
- **OpenAI GPT-4o-mini** als LLM für Antwortgenerierung
- **LangChain Memory** für Konversationskontext
- **Streamlit** als UI

---

## ⚙️ Projektstruktur

```
RAG-Hawkins/
│
├── app.py
├── init_db.py
│
├── rag/
│   ├── chat_chain.py
│   ├── embedder.py
│   ├── loader.py
│   ├── qa_chain.py
│   ├── retriever.py
│   ├── semantics.py
│   └── vectorstore.py
│
├── pdf/
│   └── (Seminarunterlagen & Prospekte)
│
└── chroma_db/
    └── (persistenter Vectorstore)
```

---

## 🧩 Dateiübersicht

### 🖥️ app.py
Streamlit-Frontend.  
- Steuert Benutzerinteraktion und Darstellung.  
- Nimmt Benutzerfragen entgegen.  
- Ruft relevante Dokumente aus der Chroma-DB ab.  
- Leitet alles an das LLM mit Conversation Memory weiter.  
- Zeigt Antwort, Verlauf und Quellen an.

### 🧱 init_db.py
Einmaliger Initialisierer der Vektordatenbank.  
- Löscht alte Chroma-DB (sauberer Neuaufbau).  
- Lädt PDF-Dateien aus `/pdf`.  
- Teilt sie in Chunks.  
- Erstellt Embeddings (OpenAI).  
- Speichert alles persistent in `chroma_db/`.

### 🧠 rag/embedder.py
Definiert das Embedding-Modell.  
- Standard: `OpenAIEmbeddings(model="text-embedding-3-large")`.  
- Dient als Schnittstelle zwischen Text und Vektorraum.

### 📄 rag/loader.py
Lädt Dokumente.  
- Liest PDF-Dateien aus einem Verzeichnis.  
- Nutzt `PyPDFLoader` aus `langchain_community`.  
- Übergibt Textsegmente an den Splitter.

### ✂️ rag/semantics.py
Verantwortlich für Textaufteilung.  
- Enthält verschiedene Split-Methoden (z. B. semantisch oder klassisch).  
- Steuert Chunkgröße und Überlappung.  
- Bereitet Daten für das Einbetten in die Chroma-DB vor.

### 🧮 rag/vectorstore.py
Schnittstelle zur Chroma-Datenbank.  
- Erstellt oder lädt persistente Datenbank.  
- Speichert Dokumente und ihre Embeddings.  
- Stellt `create_or_load_chroma()` für `init_db.py` bereit.

### 🔎 rag/retriever.py
Führt semantische Suche in der Chroma-DB aus.  
- Findet relevante Dokumentsegmente zu einer Benutzerfrage.  
- Rückgabe: Liste von LangChain-Dokumentobjekten mit `page_content` & `metadata`.

### 💬 rag/chat_chain.py
Implementiert den **Konversationsspeicher** mit `RunnableWithMessageHistory`.  
- Baut GPT-4o-Chain mit Chat-Memory auf.  
- Jede Session behält ihren Gesprächsverlauf.  
- Wird in `app.py` als Session-Objekt verwaltet.

### 🎯 rag/qa_chain.py
Steuert die **Frage-Antwort-Logik**.  
- Kombiniert Frage + Kontext (Retriever-Ergebnisse).  
- Übergibt Prompt an das LLM.  
- Extrahiert und gibt nur den Antworttext zurück.

---

## 🧠 Datenfluss (Kurz erklärt)

1. **init_db.py**  
   → PDF laden → Chunks erstellen → Embeddings erzeugen → Chroma speichern

2. **app.py**  
   → Userfrage → Retriever (Chroma) → relevante Textstellen  
   → ChatChain (mit Memory) → Antwort generieren  
   → Ausgabe: Antwort + Quellen + Verlauf

---

## 🚀 Start

```bash
# 1. Chroma-DB neu aufbauen
python init_db.py

# 2. App starten
streamlit run app.py
```

---

## ⚡ Technologie-Stack
- **Python 3.11+**
- **LangChain 0.3.x**
- **Chroma 0.4.x**
- **OpenAI GPT-4o / GPT-4o-mini**
- **Streamlit 1.40+**
- **PyPDF / BeautifulSoup / Requests**
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 08:11:16 2025

@author: milos
"""

#%% EINLEITUNG

"""
Erstellung einer Retrieval-Augmented Generation (RAG)-App

RAG (Retrieval-Augmented Generation) erweitert LLMs, indem es relevante Informationen aus externen Quellen abruft 
und in die Prompt des Modells integriert. Dadurch kann KI Fragen zu privaten oder nach dem Training 
hinzugefügten Daten beantworten und ist ideal für Q&A-Chatbots.

1️**Indexierung (Datenvorverarbeitung)**
   
   Bevor Fragen beantwortet werden können, müssen relevante Daten vorbereitet und gespeichert werden:

    - **Laden**: Dokumentlader ziehen Daten aus Quellen (Dateien, Websites usw.).
    - **Teilen**: Textsplitter zerlegen große Dokumente in kleinere Abschnitte (für bessere Suchbarkeit).
    - **Speichern**: Vektorspeicher & Einbettungen indizieren die Abschnitte zur späteren Abfrage.

2️**Abruf & Generierung (Fragen beantworten)**
   
    - **Abruf**: Durchsucht die gespeicherten Daten mittels Ähnlichkeitssuche.
    - **Generierung**: Leitet die abgerufenen Inhalte an ein LLM weiter, das eine Antwort formuliert.

**Verwendete Werkzeuge**

   - **LangChain** – Verwaltet den Abruf und das Prompt-Format.
   - **Vektordatenbanken** (ChromaDB, FAISS usw.) – Speichert und ruft Dokumenteinbettungen ab.
   - **Einbettungsmodell** (MiniLM, BGE oder OpenAI-Einbettungen) – Wandelt Text in durchsuchbare 
       Vektorrepräsentationen um.

**Warum RAG verwenden?**

Ermöglicht genaue, aktuelle Antworten auf private oder neue Daten.
Verbessert KI-Chatbots, indem Antworten auf abgerufene Informationen gestützt werden.
"""

#%% 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Bild laden und anzeigen
img = mpimg.imread("B1.png")  
plt.imshow(img)
plt.axis("off")  
plt.show()

img = mpimg.imread("B2.png")  
plt.imshow(img)
plt.axis("off")  
plt.show()

#%% 1. Indexierung: Laden

# pip install langchain langchain_community bs4
# pip install chromadb
# Visual Studio C++ erforderlich

import bs4
from langchain_community.document_loaders import WebBaseLoader

# Nur Titel, Überschriften und Inhalte des Posts aus dem HTML extrahieren.
# bs4 (BeautifulSoup) wird verwendet, um bestimmte Teile eines HTML-Dokuments zu parsen und zu extrahieren.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

# WebBaseLoader ist ein LangChain-Dokumentenlader, der Webseiten lädt und verarbeitet.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# Anzahl der Zeichen im ersten Dokument
len(docs[0].page_content)

print(docs[0].page_content[:500])

#%% 2. Indexierung: Aufteilen

"""
Das Dokument ist über 42.000 Zeichen lang – zu lang für den Kontext vieler Modelle.
Modelle haben Schwierigkeiten, Informationen in sehr langen Eingaben zu verarbeiten.
Das Dokument wird in Abschnitte aufgeteilt, eingebettet und in einem Vektorspeicher abgelegt.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Aufteilen in Abschnitte mit je 800 Zeichen und 200 Zeichen Überlappung.
# Die Überlappung verhindert, dass ein Abschnitt ohne seinen Kontext im Text gespeichert wird.
# RecursiveCharacterTextSplitter nutzt Trennzeichen wie Zeilenumbrüche zur Aufteilung.
# add_start_index=True – fügt jedem Chunk einen Index hinzu, um seine Position im Originaltext zu verfolgen.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, # max
    chunk_overlap=200, 
    add_start_index=True)

all_splits = text_splitter.split_documents(docs)

# Gesamtanzahl der Chunks
len(all_splits)

# Der erste Chunk hat 969 Zeichen (etwas weniger als das Limit von 1000 Zeichen)
len(all_splits[0].page_content)

# Metadaten des 11. Chunks
all_splits[10].metadata

#%% 3. Indexierung: Speichern

"""
66 Textabschnitte werden indexiert. 
Der Inhalt jedes Chunks wird eingebettet und die resultierenden Embeddings 
werden in einer Vektor-Datenbank gespeichert.
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Embedding-Modell 
local_embeddings = OllamaEmbeddings(model="all-minilm:33m")

# Vektor-Datenbank Chroma mit lokalen Embeddings erstellen
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)


#%% 4. Abruf & Generierung: Abrufen

"""
Anwendung zur Beantwortung von Nutzerfragen mit Vektorbasierter Suche

1. Nutzerfrage
    Für die Suche wird die Nutzerfrage ebenfalls eingebettet. 

2. Dokumentensuche:
    VectorStoreRetriever. Hierbei werden sowohl die Dokumente als auch die Nutzerfrage in Vektoren 
    (Embeddings) umgewandelt. Die Suche erfolgt dann über die Berechnung der Ähnlichkeit 
    zwischen diesen Vektoren, meist mittels Cosinus-Ähnlichkeit - Winkel zwischen zwei Vektoren.
    Je kleiner der Winkel, desto größer ist die Ähnlichkeit der Inhalte.
    So können auch inhaltlich ähnliche, aber nicht exakt gleiche Formulierungen erkannt werden.

3. Antwortgenerierung:
    Die gefundenen relevanten Dokumente sowie die ursprüngliche Frage werden 
    gemeinsam an ein Sprachmodell übergeben. Das Modell analysiert die Inhalte und 
    erstellt daraus eine präzise Antwort für den Nutzer.
"""

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

# Findet die 6 relevantesten Chunks aus den gespeicherten Dokumenten
len(retrieved_docs)

for i, doc in enumerate(retrieved_docs):
    print(f"Chunk {i+1}:\n{doc.page_content}\n")
    
#%% 5. Abruf & Generierung: Generieren

# Eine Chain, die eine Frage entgegennimmt, relevante Dokumente abruft,
# einen Prompt erstellt, diesen an ein Modell übergibt und die Ausgabe verarbeitet.

from langchain_ollama import OllamaLLM
from langsmith import Client

# Ollama-LLM laden
llm = OllamaLLM(model="llama3.1:latest")

# LangSmith Client erstellen
client = Client()

# Vorgefertigter Prompt für Retrieval-Augmented Generation (RAG)
prompt = client.pull_prompt("rlm/rag-prompt")

# Füllt die Vorlage mit dem angegebenen "context" und der "question"
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

print(example_messages[0].content)


#%% Retrieval-Augmented Generation (RAG) Chain
"""
Nutzerfrage eingeben: Die Anwendung erhält eine Frage.

Dokumente suchen: Der Retriever findet relevante Textabschnitte, die formatiert werden.

Prompt erstellen: Die Frage und der Kontext werden in ein Prompt-Template eingefügt.

Antwort generieren: Das Sprachmodell erstellt eine Antwort.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Dokumente formatieren, damit sie lesbar sind
def format_docs(docs):
    """
    Verkettet den Inhalt (page_content) jedes Dokuments, getrennt durch zwei Zeilenumbrüche.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain erstellen
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
"""
1. {"context": retriever | format_docs, "question": RunnablePassthrough()}
    Dictionary mit zwei Einträgen:
        "context": Die Nutzerfrage wird zunächst an den retriever. 
        Diese werden anschließend mit format_docs formatiert.
        "question": Die ursprüngliche Frage wird unverändert durchgereicht.

2. Das Dictionary wird an ein Prompt-Template (prompt) übergeben, das die Frage und 
    den Kontext in ein geeignetes Format für das Sprachmodell bringt.

3. Das Sprachmodell (llm) generiert eine Antwort basierend auf dem Prompt.

4. Die Roh-Ausgabe des Modells wird mit StrOutputParser() in einen lesbaren String umgewandelt.
"""

# Antwort zeichenweise ausgeben
for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)
    
"""
rag_chain.stream(...):
Startet die Kette mit der Frage "What is Task Decomposition?" und gibt die Antwort als Stream zurück.

for chunk in ...:
Die Antwort wird Stück für Stück (z.B. zeichenweise oder satzweise) ausgegeben.

print(chunk, end="", flush=True):
Gibt jedes empfangene Chunk direkt aus, ohne Zeilenumbruch. 
Mit flush=True wird sichergestellt, dass die Ausgabe sofort sichtbar ist (wichtig für Live-Feedback).
"""



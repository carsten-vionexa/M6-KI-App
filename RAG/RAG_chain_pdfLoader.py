# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 13:03:40 2025

@author: milos
"""

# --- 1. Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --- 2. Load PDF File ---
file_path = "C:\\Users\\milos\\Desktop\\DATA\\Educx\\Modul_6_KI\\KI_Lernplan\\KI_W2_T6\\RAG_chain\\nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()


# --- 3. Inspect Loaded Documents (Optional, for debugging) ---
print(len(docs))
print(docs[1].page_content[0:1000])
print(docs[1].metadata)


# --- 4. Split Documents into Chunks ---
# Chunks of 1000 characters, with 200-character overlap for context continuity
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# --- 5. Create Embeddings for Chunks ---
# Use a local MiniLM model via Ollama for embedding text chunks
local_embeddings = OllamaEmbeddings(model="all-minilm:33m")


# --- 6. Build a Chroma Vector Store ---
# Store all text chunks as vectors for efficient similarity search
vectorstore = Chroma.from_documents(documents=splits, embedding=local_embeddings)


# --- 7. Define the System Prompt for the LLM ---
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


# --- 8. Create a Prompt Template ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# --- 9. Load the Language Model and Setup Retriever ---
# Use Llama 3.1 via Ollama for answer generation
llm = OllamaLLM(model="llama3.1:latest")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


# --- 10. Build the RAG Chain ---
# Create a chain that combines retrieval and answer generation

# Dokumente formatieren, damit sie lesbar sind
def format_docs(docs):
    """
    Verkettet den Inhalt (page_content) jedes Dokuments, getrennt durch zwei Zeilenumbrüche.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain erstellen
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What was Nike's revenue in 2023?")

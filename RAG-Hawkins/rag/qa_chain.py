from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List

SYSTEM_PROMPT = """Du beantwortest ausschließlich Fragen aus dem gegebenen Kontext.
Wenn die Antwort nicht eindeutig im Kontext steht, schreibe: "Ich weiß es nicht."
"""

def answer_question(query: str, docs: List[Document], model_name="gpt-4o-mini"):
    context = "\n\n".join(
        f"[{d.metadata.get('source')}, Seite {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT + "\n\nFrage: {question}\n\nKontext:\n{context}\n\nAntwort:",
        input_variables=["question", "context"],
    )
    llm = ChatOpenAI(model=model_name, temperature=0)
    return llm.invoke(prompt.format(question=query, context=context))
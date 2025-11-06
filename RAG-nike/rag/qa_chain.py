# rag/qa_chain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

QA_SYSTEM = (
    "Du beantwortest NUR aus dem Kontext. Wenn die Antwort nicht im Kontext steht, "
    "antworte: \"Ich weiß es nicht.\""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        ("human", "Frage:\n{question}\n\nKontext:\n{context}")
    ]
)

def answer_question(question: str, context: str, model: str = "gpt-4o-mini") -> str:
    llm = ChatOpenAI(model=model, temperature=0)
    chain = prompt | llm
    out = chain.invoke({"question": question, "context": context})
    return getattr(out, "content", str(out))
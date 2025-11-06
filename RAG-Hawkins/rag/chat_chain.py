# rag/chat_chain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Speicher für mehrere Sessions
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def create_chat_chain(model_name="gpt-4o-mini"):
    prompt = ChatPromptTemplate.from_template(
        """Bisheriges Gespräch:
{history}

Neue Frage:
{input}

Antworte auf Basis der bisherigen Unterhaltung und des gegebenen Kontexts."""
    )

    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
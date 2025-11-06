# rag/chat_chain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# In-Memory Chat-Histories per Session-ID
_histories: dict[str, InMemoryChatMessageHistory] = {}

def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]


SYSTEM = (
    "Du bist ein präziser Assistent für einen RAG über den Nike 10-K Bericht.\n"
    "Beantworte NUR anhand des bereitgestellten Kontextes.\n"
    "Wenn etwas nicht eindeutig im Kontext steht, antworte: 'Ich weiß es nicht.'"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("history"),
        ("human",
         "Frage:\n{question}\n\n"
         "Kontext (mit Quellenmarkern in eckigen Klammern):\n{context}\n\n"
         "Antworte kurz, sachlich und ohne Floskeln.")
    ]
)


def create_chat_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = prompt | llm

    # robust gegen unterschiedliche Formate von config
    def get_session_history(config):
        if isinstance(config, dict):
            session_id = (
                config.get("configurable", {}).get("session_id", "default-session")
            )
        else:
            session_id = "default-session"
        return _get_history(session_id)

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )
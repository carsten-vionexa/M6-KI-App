
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import importlib
import sys
sys.modules["litellm"] = importlib.import_module("litellm")

# 🔑 API Key
os.environ["SERPER_API_KEY"] = "eaa5fbce8884faccebb1832983a6839d8472da29"


# 🔄 Lokales Modell
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# 🔍 Tool zum Suchen
search_tool = SerperDevTool()

# 🧩 Agent 1: Recherche
search_agent = Agent(
    role="Web Researcher",
    goal="Find the most recent news about a given topic",
    backstory="An AI research assistant with expertise in real-time web searches and news analysis.",
    tools=[search_tool],
    verbose=True,
    memory=True,
    llm=local_llm
)

# 🧩 Agent 2: Schreibkraft
writer_agent = Agent(
    role="News Analyst",
    goal="Write a clean, concise summary of the recent news",
    backstory="A clear communicator who distills complex news topics into easy-to-understand summaries.",
    verbose=True,
    memory=True,
    llm=local_llm
)

# 🗂 Task 1: Suche
search_task = Task(
    description="Search the web for the latest news related to {topic}. Return key facts and headlines.",
    expected_output="A list of 5-10 relevant and recent headlines with 1-2 sentence summaries for each.",
    agent=search_agent
)

# 🗂 Task 2: Zusammenfassen
summary_task = Task(
    description="Using the results from the search, write a well-structured summary report.",
    expected_output="A 3-paragraph summary of the most recent developments related to {topic}.",
    agent=writer_agent
)

# 🚀 Crew
crew = Crew(
    agents=[search_agent, writer_agent],
    tasks=[search_task, summary_task],
    process=Process.sequential
)

result = crew.kickoff(inputs={"topic": "AI regulation in Europe"})
print(result)
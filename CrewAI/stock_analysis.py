# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:04:45 2025

@author: milos
"""

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# =====================
# Environment Variables
# =====================
# Make sure to set your keys before running:
# export SERPER_API_KEY="your_serper_api_key"
# export OPENAI_API_KEY="your_openai_api_key"

os.environ["SERPER_API_KEY"] = ""

# Use local Ollama model, e.g., 'llama3'
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# =====================
# Tools
# =====================
search_tool = SerperDevTool()

# =====================
# Agents
# =====================
market_researcher = Agent(
    role="Market Researcher",
    goal="Gather up-to-date financial and market information about {ticker}.",
    backstory=(
        "You are an expert financial researcher. "
        "You can analyze company performance, identify key trends, "
        "and interpret recent market data for investors."
    ),
    tools=[search_tool],
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM

)

sentiment_analyst = Agent(
    role="Sentiment Analyst",
    goal="Assess investor and media sentiment from recent news about {ticker}.",
    backstory=(
        "You are a skilled sentiment analyst who studies market news and "
        "social tone to identify overall investor confidence and risk perception."
    ),
    tools=[search_tool],
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM

)

investment_writer = Agent(
    role="Investment Report Writer",
    goal="Summarize financial and sentiment insights into a clear investment report for {ticker}.",
    backstory=(
        "You are a talented financial journalist who transforms raw data and "
        "analysis into actionable, professional insights for investors."
    ),
    verbose=True,
    memory=True,
    allow_delegation=False,
    llm=local_llm  # <-- use local LLM

)

# =====================
# Tasks
# =====================
research_task = Task(
    description=(
        "Collect the latest financial data and market performance summary "
        "for the company identified by the ticker symbol {ticker}. "
        "Focus on quarterly performance, market positioning, and key developments."
    ),
    expected_output=(
        "A detailed summary of {ticker}'s latest financial highlights, "
        "recent performance trends, and overall market standing."
    ),
    tools=[search_tool],
    agent=market_researcher
)

sentiment_task = Task(
    description=(
        "Analyze recent news and media sentiment surrounding {ticker}. "
        "Identify whether the overall tone is positive, neutral, or negative, "
        "and provide examples of key stories influencing this sentiment."
    ),
    expected_output=(
        "A concise sentiment overview of {ticker}, summarizing market perception "
        "and notable positive or negative trends in the news."
    ),
    tools=[search_tool],
    agent=sentiment_analyst
)

report_task = Task(
    description=(
        "Combine the insights from financial research and sentiment analysis "
        "to create a cohesive investment insight report. "
        "Summarize key opportunities, risks, and your overall market outlook for {ticker}."
    ),
    expected_output=(
        "A 3–4 paragraph investment insight report for {ticker}, "
        "including major opportunities, risks, and your general recommendation."
    ),
    agent=investment_writer,
    async_execution=False,
    output_file="investment_report.md"
)

# =====================
# Crew Definition
# =====================
crew = Crew(
    agents=[market_researcher, sentiment_analyst, investment_writer],
    tasks=[research_task, sentiment_task, report_task],
    process=Process.sequential
)

# =====================
# Execution
# =====================
if __name__ == "__main__":
    result = crew.kickoff(inputs={"ticker": "AAPL"})
    print("\n=== FINAL INVESTMENT REPORT ===\n")
    print(result)

y# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:54:53 2025

@author: milos
"""

#%% Crew 1

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Serper key still needed for the search tool
os.environ["SERPER_API_KEY"] = "c26fca71fd76512a0f1ff1416e377519522d4af4"

# Use local Ollama model, e.g., 'llama3'
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# llama3.1:latest 

# Tool to search the web
search_tool = SerperDevTool()

# Agent that performs the search
search_agent = Agent(
    role="Web Researcher",
    goal="Find the most recent news about a given topic",
    backstory="An AI research assistant with expertise in real-time web searches and news analysis.",
    tools=[search_tool],
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Agent that writes a summary
writer_agent = Agent(
    
    role="News Analyst",
    
    # The agent's *goal* is the core objective it tries to achieve during its assigned tasks.
    # This helps the LLM stay focused on what matters most for this agent.
    goal="Write a clean, concise summary of the recent news",
    
    # The agent's *backstory* provides contextual depth to the agent.
    # This doesn't affect logic but significantly improves the LLM’s responses by grounding them in a persona.
    backstory="A clear communicator who distills complex news topics into easy-to-understand summaries.",
    verbose=True,
    
    # Whether or not the agent retains memory of prior interactions.
    # If set to True, the agent can use context from earlier in the run, helpful for coherent multi-step tasks.
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Task to perform the search
search_task = Task(
    description="Search the web for the latest news related to {topic}. Return key facts and headlines.",
    expected_output="A list of 5-10 relevant and recent headlines with 1-2 sentence summaries for each.",
    agent=search_agent
)

# Task to write the summary
summary_task = Task(
    description="Using the results from the search, write a well-structured summary report.",
    expected_output="A 3-paragraph summary of the most recent developments related to {topic}.",
    agent=writer_agent
)

# Create the crew
crew = Crew(
    agents=[search_agent, writer_agent],
    tasks=[search_task, summary_task],
    process=Process.sequential
)

# Run the crew
result = crew.kickoff(inputs={"topic": "AI regulation in Europe"})
print(result)

#%% Crew 2

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool

# Use local Ollama model, e.g., 'llama3'
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# Tool to scrape website content
scrape_tool = ScrapeWebsiteTool()

# Agent to scrape the website
scraper_agent = Agent(
    role="Web Scraper",
    goal="Extract raw content from the specified website URL",
    backstory="An expert at navigating websites and collecting data from them.",
    tools=[scrape_tool],
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Agent to analyze and summarize the content
analyzer_agent = Agent(
    role="Content Analyst",
    goal="Summarize the content extracted from a website into an insightful report",
    backstory="A skilled analyst who turns raw data into understandable insights.",
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Task for scraping
scrape_task = Task(
    description="Scrape all the readable content from the website {url}.",
    expected_output="Raw website text content scraped from {url}.",
    agent=scraper_agent,
)

# Task for summarizing
summarize_task = Task(
    description="Take the scraped content from the website and summarize it into a clear, informative report.",
    expected_output="A detailed summary of the content from {url}, covering main topics and highlights.",
    agent=analyzer_agent,
)

# Create the crew
crew = Crew(
    agents=[scraper_agent, analyzer_agent],
    tasks=[scrape_task, summarize_task],
    process=Process.sequential
)

# Kick it off
result = crew.kickoff(inputs={"url": "https://lilianweng.github.io/posts/2023-06-23-agent/"})
print(result)

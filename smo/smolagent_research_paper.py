# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:06:28 2025

@author: milos
"""
# pip install arxiv pypdf

from smolagents import tool 
import requests
from bs4 import BeautifulSoup
import json
from huggingface_hub import HfApi
import arxiv
from pypdf import PdfReader

@tool
def get_hugging_face_top_daily_paper() -> str:
    """
    This is a tool that returns the most upvoted paper on Hugging Face daily papers.
    It returns the title of the paper
    """
    try:
      url = "https://huggingface.co/papers"
      response = requests.get(url)
      response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
      
      # Parse the received HTML content using BeautifulSoup
      soup = BeautifulSoup(response.content, "html.parser")
      
      # Find all <div> elements with specific classes indicating they may contain relevant data
      # Extract the title element from the JSON-like data in the "data-props" attribute
      containers = soup.find_all('div', class_='SVELTE_HYDRATER contents')
      top_paper = ""

      for container in containers:
          data_props = container.get('data-props', '')
          if data_props:
              try:
                  # Parse the JSON-like string
                  json_data = json.loads(data_props.replace('&quot;', '"'))
                  if 'dailyPapers' in json_data:
                      top_paper = json_data['dailyPapers'][0]['title']
              except json.JSONDecodeError:
                  continue

      return top_paper
    except requests.exceptions.RequestException as e:
      print(f"Error occurred while fetching the HTML: {e}")
      return None

print(get_hugging_face_top_daily_paper())




@tool
def get_paper_id_by_title(title: str) -> str:
    """
    This is a tool that returns the arxiv paper id by its title.
    It returns the title of the paper

    Args:
        title: The paper title for which to get the id.
    """
    api = HfApi()
    papers = api.list_papers(query=title) # paper search
    
    # If papers not empty return first paper's id
    if papers:
        paper = next(iter(papers))
        return paper.id
    else:
        return None
    
print(get_paper_id_by_title('InteractComp: Evaluating Search Agents With Ambiguous Queries'))

    

@tool
def download_paper_by_id(paper_id: str) -> None:
    """
    This tool gets the ID of a paper and downloads it from arXiv.
    It saves the paper locally in the current directory as "paper.pdf".

    Args:
        paper_id: The arXiv ID of the paper to download (e.g., "1706.03762").
    """
    
    # Create a new client instance to interact with arXiv.
    # Perform a search using the given arXiv ID.
    # id_list must be a list, even when it contains only one ID.
    # results() returns a generator, so we extract the first paper result using next().
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
    paper.download_pdf(filename="paper.pdf")
    return None

download_paper_by_id("2510.24668")


@tool
def read_pdf_file(file_path: str) -> str:
    """
    This function reads the first three pages of a PDF file and returns its content as a string.
    Args:
        file_path: The path to the PDF file.
    Returns:
        A string containing the content of the PDF file.
    """
    content = ""
    reader = PdfReader('paper.pdf')
    print(len(reader.pages))
    pages = reader.pages[:3]
    for page in pages:
        content += page.extract_text()
    return content

text = read_pdf_file('paper.pdf')

#%%

from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="ollama_chat/qwen3-coder:480b-cloud")

agent = CodeAgent(tools=[get_hugging_face_top_daily_paper,
                         get_paper_id_by_title,
                         download_paper_by_id,
                         read_pdf_file],
                  model=model,
                  stream_outputs=True)

agent.run(
    "Summarize today's top paper on Hugging Face daily papers.",
)




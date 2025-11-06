# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 18:08:48 2025

@author: milos
"""

import datasets
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Load the Hugging Face documentation dataset
# This dataset contains contents of Hugging Face documentation pages
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Filter to include only Transformers documentation
# Each row has a "source" field, e.g., "huggingface/transformers/..."
# We only want rows starting with "huggingface/transformers"
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))
print(knowledge_base[0])

# Convert dataset entries to Document objects with metadata
# Each document contains the actual text and its associated metadata (e.g., 'transformers' as source)
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    add_start_index=True, # Adds starting character index metadata for chunks
    strip_whitespace=True, 
    separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")

#%%

from smolagents import Tool

# Define a custom tool that inherits from the base Tool class
class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    # Initialization method where we pass in our chunked documents
    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # Initialize the retriever with our processed documents
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # Return top 10 most relevant documents
        )
        # Keyword overlap - doesn't require embeddings or external models — uses raw text.
        
    # Main method called when this tool is used — performs the retrieval logic
    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"

        # Use the retriever's `invoke()` method to fetch the top-k relevant documents
        docs = self.retriever.invoke(query)

        # Format the retrieved documents for readability
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize our retriever tool with the processed documents
retriever_tool = RetrieverTool(docs_processed)

# Run a query
retriever_tool.forward("For a transformers model training, which is slower, the forward or the backward pass?")

#%%

from smolagents import LiteLLMModel, CodeAgent

model = LiteLLMModel(model_id="ollama_chat/qwen3-coder:480b-cloud")

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],  # List of tools available to the agent
    model=model,  
    max_steps=4,  # Limit the number of reasoning steps
    verbosity_level=2,  # Step-by-step details of decision-making, thoughts, and tool usage
)

#%%

# Ask a question that requires retrieving information
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# Run the agent to get an answer
agent_output = agent.run(question)

# Display the final answer
print("\nFinal answer:")
print(agent_output)

"""
Out - Final answer: backward
[Step 2: Duration 1.73 seconds| Input tokens: 5,957 | Output tokens: 197]

Final answer:
backward
"""
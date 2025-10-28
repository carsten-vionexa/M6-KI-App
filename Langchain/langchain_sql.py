# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 10:21:40 2025

@author: milos
"""

#%% Build a Question/Answering system over SQL data

"""
At a high-level, the steps are:

Convert question to SQL query: Model converts user input to a SQL query.
Execute SQL query: Execute the query.
Answer the question: Model responds to user input using the query results.
"""

#%%

from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

# pip install langchain langchain-community langgraph psycopg psycopg[binary]

#%% Connect to database

# Establish a connection to the PostgreSQL 'chinook' database using provided credentials.
db = SQLDatabase.from_uri("postgresql+psycopg://postgres:carsten25@localhost:5432/chinook1")

# This helps confirm the type of database and SQL syntax that will be used for queries.
print(db.dialect)

# Retrieve and print a list of table names that are usable in the current database.
print(db.get_usable_table_names())

# Run a SQL query to select the first 10 rows from the 'Artist' table and print the results.
print(db.run("SELECT * FROM Artist LIMIT 10;"))

#%% Define Langgraph state

# The LangGraph state of our application controls what data is input to the application, 
# transferred between steps, and output by the application.

# For this application, we can just keep track of the input question, generated query, 
# query result, and generated answer:

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

#%%

# Import Ollama model
model = ChatOllama(model="llama3.1")

# Define prompt
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

for message in query_prompt_template.messages:
    message.pretty_print()
    
#%% 1. Generating the SQL Query

# This is used here to provide a description for the LLM.
from typing_extensions import Annotated
# TypedDict is used to define the structure of a dictionary with type hints.
from typing import TypedDict

# Define the structure for the LLM's output.
# By using TypedDict, we tell the LLM to generate a JSON object
# with a specific key ("query") and a string value.
class QueryOutput(TypedDict):
    """Generated SQL query."""

    # The 'query' field must be a string.
    # The Annotated metadata provides a description to the LLM, guiding it
    # to generate a "Syntactically valid SQL query."
    query: Annotated[str, "A description for the LLM.", "Syntactically valid SQL query."]


# This function represents a node in your LangGraph graph.
# It takes the current state of the graph as input.
def write_query(state: State):
    """Generate SQL query to fetch information."""

    # 1. Create the prompt for the LLM.
    # The prompt template is filled with context about the database (dialect, table info)
    # and the user's question, which is retrieved from the input state.
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )

    # 2. Configure the LLM for structured output.
    # .with_structured_output() forces the LLM to return a JSON object
    # that strictly conforms to the structure of the QueryOutput class.
    # This makes the output reliable and easy to parse.
    structured_llm = model.with_structured_output(QueryOutput)

    # 3. Invoke the LLM.
    # The LLM processes the prompt and generates the structured output.
    result = structured_llm.invoke(prompt)

    # 4. Update the state.
    # The function returns a dictionary to update the graph's state.
    # The generated SQL query is added to the state under the key "query".
    # This makes it available for the next node in the graph.
    return {"query": result["query"]}

write_query({"question": "Wie viele Mitarbeiter gibt es"})

#%% 2. Executing the SQL Query

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

# This function is another node in the graph.
def execute_query(state: State):
    """Execute SQL query."""
    
    # 1. Initialize the SQL execution tool.
    # The QuerySQLDatabaseTool requires a database connection object (db).
    execute_query_tool = QuerySQLDatabaseTool(db=db)

    # 2. Execute the query from the state.
    # The tool's .invoke() method takes the SQL string as input.
    # The query is retrieved from the state, where it was placed by the `write_query` node.
    # The tool runs the query and captures the output (e.g., a list of results or an error).
    query_result = execute_query_tool.invoke(state["query"])

    # 3. Update the state with the result.
    # The function returns a dictionary to update the graph's state.
    # The result from the database is added to the state under the key "result".
    return {"result": query_result}

# It simulates a state where the "query" key already exists,
# allowing you to test the execution logic independently.
execute_query({"query": "SELECT COUNT(*) AS EmployeeCount FROM Employee;"})


#%% 3. Generate answer

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = model.invoke(prompt)
    return {"answer": response.content}

#%% Compile app using langgraph 

from langgraph.graph import START, StateGraph
from IPython.display import Image, display

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)

graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# Image
display(Image(graph.get_graph().draw_mermaid_png()))

for step in graph.stream(
    {"question": "How many employees are there?"}, stream_mode="updates"
):
    print(step)
    
#%% Human-in-the-loop

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# Now that we're using persistence, we need to specify a thread ID
# so that we can continue the run after review.
config = {"configurable": {"thread_id": "1"}}

display(Image(graph.get_graph().draw_mermaid_png()))

for step in graph.stream(
    {"question": "How many employees are there?"},
    config,
    stream_mode="updates",
):
    print(step)

try:
    user_approval = input("Do you want to go to execute query? (yes/no): ")
except Exception:
    user_approval = "no"

if user_approval.lower() == "yes":
    # If approved, continue the graph execution
    for step in graph.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:10:23 2025

@author: milos
"""

#%% CodeAgent

# pip install smolagents

from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, LiteLLMModel

# model = InferenceClientModel(model_id='Qwen/Qwen2.5-Coder-32B-Instruct', token="")
# model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

model = LiteLLMModel(model_id="ollama_chat/qwen3-coder:480b-cloud")

agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

"""
Out - Final answer: 9.931034482758621
[Step 1: Duration 15.88 seconds| Input tokens: 2,018 | Output tokens: 332]
Out[3]: "\nOut - Final answer: 9\n[Step 7: Duration 2.22 seconds| Input tokens: 22,125 | Output tokens: 797]\nOut[1]: '\nOut - Final answer: 9.620689655172415\n[Step 2: Duration 3.32 seconds| Input tokens: 6,457 | Output tokens: 315]\nOut[12]: 9.620689655172415\n'"
"""


#%% ToolCallingAgent

from smolagents import ToolCallingAgent, LiteLLMModel

model = LiteLLMModel(model_id="ollama_chat/qwen3-coder:480b-cloud")

agent = ToolCallingAgent(tools=[WebSearchTool()], model=model)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

"""
[Step 20: Duration 2.66 seconds| Input tokens: 242,130 | Output tokens: 790]
Reached max steps.
[Step 21: Duration 4.99 seconds| Input tokens: 265,205 | Output tokens: 1,068]
Out[11]: "To determine how many seconds it would take for a leopard at full speed to run through 
Pont des Arts, we need to use the length of the bridge and the top speed of a leopard.\n\nFrom 
the search results, we know:\n- The length of Pont des Arts is 155 meters.\n- The top speed of a 
leopard is approximately 58 km/h (36 mph).\n\nFirst, we need to convert the leopard's speed from 
km/h to meters per second (m/s) for consistency in units:\n\\[ 58 \\, \\text{km/h} = \\frac{58 \\times 1000 \\, \\text{meters}}{3600 \\, \\text{seconds}} 
                                                              \\approx 16.11 \\, \\text{m/s} \\]\n\nNext, we use the formula:\n\\[ \\text{Time} = \\frac{\\text{Distance}}{\\text{Speed}} \\]\n\nSubstituting the values:\n\\[ \\text{Time} = \\frac{155 \\, \\text{meters}}{16.11 \\, \\text{m/s}} \\approx 9.62 \\, \\text{seconds} \\]
\n\nTherefore, it would take a leopard approximately 9.62 seconds to run through Pont des Arts at full speed."
"""

#%% CodeAgent

from smolagents import CodeAgent, WebSearchTool, LiteLLMModel

model = LiteLLMModel(
        model_id="ollama_chat/qwen3-coder:480b-cloud",
    )

agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

agent.run("How does DeepSeek-v3 compare to DeepSeek-v2? What is the percent increase or decrease from one to the other?")

"""
'DeepSeek-V3 shows significant improvements over DeepSeek-V2, with 184.32% more total 
parameters (671B vs 236B), enhanced architecture, and reportedly lower training costs 
($6 million). Both models use Mixture-of-Experts architecture, but V3 offers better 
efficiency and performance.'
"""

#%% Custom tools

from smolagents import tool

@tool
def get_weather_by_city(city: str) -> str:
    """A tool that returns the current weather for a given city.
    Args:
        city: Name of the city to get weather information for.
    """
    # Simulated output — Replace this with real API integration if desired.
    mock_weather = {
        "New York": "Partly cloudy, 22°C",
        "London": "Rainy, 16°C",
        "Tokyo": "Sunny, 27°C",
        "Paris": "Overcast, 19°C"
    }

    weather = mock_weather.get(city, "Weather data not available for this city.")
    return f"The current weather in {city} is: {weather}"

model = LiteLLMModel(
        model_id="ollama_chat/qwen2.5-coder:7b",
    )

agent = CodeAgent(tools=[get_weather_by_city], model=model, stream_outputs=True)

agent.run("What's the weather like in Tokyo?")



